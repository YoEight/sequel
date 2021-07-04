use async_stream::stream;
use futures::stream::BoxStream;
use futures::TryStreamExt;
use sqlparser::ast::{BinaryOperator, Expr, UnaryOperator};

use crate::types::{self, Either, Error, Line, Op, QueryInfo, Register, Suspension, Value};

fn stack_pop<A>(stack: &mut Vec<A>) -> crate::Result<A> {
    match stack.pop() {
        Some(op) => Ok(op),
        None => Error::failure("Unexpected empty stack"),
    }
}

pub async fn run<'a, S>(
    source: S,
    statement: sqlparser::ast::Statement,
) -> crate::Result<BoxStream<'a, crate::Result<types::Line>>>
where
    S: types::DataSource + Send + Sync + 'a,
{
    let stream = async_stream::try_stream! {
        match statement {
            sqlparser::ast::Statement::Query(query) => {
                let info = types::collect_query_info(&query)?;
                let mut sub_queries_register = std::collections::HashMap::new();
                let mut env = types::Env::new();
                let mut offset = 0usize;

                for name in info.sub_queries() {
                    let sub_query_vec: Vec<types::Line> = source.fetch(name)
                        .await?
                        .try_collect()
                        .await?;

                    sub_queries_register.insert(name.clone(), sub_query_vec);
                }

                if let sqlparser::ast::SetExpr::Select(select) = query.body {
                    if select.from.is_empty() {
                        let mut line = std::collections::HashMap::new();

                        for item in select.projection {
                            match item {
                                sqlparser::ast::SelectItem::UnnamedExpr(expr) => {
                                    let value = evaluate(&mut env, &info, &sub_queries_register, &expr)?.into_right().unwrap();
                                    line.insert(offset.to_string(), value);

                                    offset += 1;
                                }

                                sqlparser::ast::SelectItem::ExprWithAlias { expr, alias } => {
                                    let value = evaluate(&mut env, &info, &sub_queries_register, &expr)?.into_right().unwrap();
                                    line.insert(alias.value, value);
                                }

                                _ => {}
                            }
                        }

                        yield line;
                    } else {
                        if let Some(main_source) = info.source_name.as_ref() {
                            let mut env = types::Env::new();
                            let mut joins_vecs = Vec::new();

                            // We have no other choice that loading all joined tables in memory.
                            // TODO - There are probably better ways to achieve this but those are
                            // out of my reach right now.
                            for join in main_source.joins.iter() {
                                let vs: Vec<types::Line> = source.fetch(&join.source_name).await?.try_collect().await?;
                                joins_vecs.push((join, vs));
                            }

                            // It means we need to load the main source table entirely in memory.
                            if info.contains_right_join() {
                                let main_source_vec: Vec<types::Line> = source.fetch(&main_source).await?.try_collect().await?;
                            } else {
                                // it begs the question if it's actually useful to do that, considering only
                                // simple queries would benefit it.
                                let mut main_source_stream = source.fetch(&main_source).await?;

                                while let Some(mut line) = main_source_stream.try_next().await? {
                                    println!("{:?} --->> {:?}", main_source.name, line);
                                    let mut skip = false;
                                    line = types::rename_line(&main_source, line);
                                    env.enter_scope(line);

                                    for (join, join_vec) in joins_vecs.iter() {
                                        match join.r#type {
                                            types::JoinType::Inner => {
                                                let mut matched = false;

                                                for join_line in join_vec.iter() {
                                                    println!("{:?} <<--- {:?}", join.source_name.name, join_line);
                                                    println!("Before merge Env: {:?}", env);
                                                    env.merge_scope(types::rename_line(&join.source_name, join_line.clone()));

                                                    println!("After merge Env: {:?}", env);
                                                    if let Some(expr) = join.expr.as_ref() {
                                                        // We can't end up in a suspension use-case because not possible
                                                        // when joining tables.
                                                        if let types::Either::Right(value) = evaluate(&mut env, &info, &sub_queries_register, expr)? {
                                                            println!("Value === {:?}", value);
                                                            if value.as_bool()? {
                                                                matched = true;
                                                                break;
                                                            }
                                                        }
                                                    }
                                                    env.exit_scope();
                                                }

                                                skip = !matched;
                                            }
                                            types::JoinType::Left => {
                                                for join_line in join_vec.iter() {
                                                    env.merge_scope(types::rename_line(&join.source_name, join_line.clone()));

                                                    if let Some(expr) = join.expr.as_ref() {
                                                        // We can't end up in a suspension use-case because not possible
                                                        // when joining tables.
                                                        if let types::Either::Right(value) = evaluate(&mut env, &info, &sub_queries_register, expr)? {
                                                            if value.is_null() || !value.as_bool()? {
                                                                env.exit_scope();

                                                                break;
                                                            }
                                                        }
                                                    }
                                                }
                                            }

                                            // Not possible because we already pre-checked we are not a right or full join,
                                            _ => unreachable!(),
                                        }

                                        if skip {
                                            break;
                                        }
                                    }

                                    // Where evaluation loop.
                                    if let Some(expr) = info.selection.as_ref() {
                                        loop {
                                            match evaluate(&mut env, &info, &sub_queries_register, expr)? {
                                                Either::Left(susp) => {

                                                }
                                                Either::Right(value) => {
                                                    if value.is_null() || !value.as_bool()? {
                                                        env.exit_scope();
                                                        skip = true;

                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    if skip {
                                        continue;
                                    }

                                    let line = env.project_line(&info)?;

                                    yield line;

                                    env.exit_scope();
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    };

    Ok(Box::pin(stream))
}

pub fn evaluate<'a>(
    env: &mut types::Env,
    main_info: &QueryInfo,
    sub_query_register: &Register,
    expr: &'a Expr,
) -> crate::Result<Either<Suspension<'a>, types::Value>> {
    let mut params = Vec::<types::Param>::new();
    let mut stack = Vec::<Op<'a>>::new();
    let mut result = None;
    let empty_line: Line = Default::default();

    stack.push(Op::Expr(expr));

    loop {
        if let Some(expr) = stack.pop() {
            match expr {
                Op::Return => {
                    if !env.exit_scope() {
                        break;
                    }
                }

                Op::Binary(op) => {
                    let left = stack_pop(&mut params)?.as_value()?;
                    let right = stack_pop(&mut params)?.as_value()?;

                    if left.is_null() && right.is_null() && op == BinaryOperator::Spaceship {
                        stack.push(Op::Value(Value::Bool(true)));

                        continue;
                    }

                    if (left.is_null() || right.is_null()) && op == BinaryOperator::Spaceship {
                        stack.push(Op::Value(Value::Bool(true)));

                        continue;
                    }

                    if left.is_null() || right.is_null() {
                        stack.push(Op::Value(Value::Null));

                        continue;
                    }

                    let result = match (left, right, op) {
                        (Value::Number(left), Value::Number(right), BinaryOperator::Plus) => {
                            Ok(Value::Number(left + right))
                        }
                        (Value::Number(left), Value::Number(right), BinaryOperator::Minus) => {
                            Ok(Value::Number(left - right))
                        }
                        (Value::Number(left), Value::Number(right), BinaryOperator::Multiply) => {
                            Ok(Value::Number(left * right))
                        }
                        (Value::Number(left), Value::Number(right), BinaryOperator::Modulus) => {
                            Ok(Value::Number(left % right))
                        }
                        (Value::Number(left), Value::Number(right), BinaryOperator::BitwiseAnd) => {
                            Ok(Value::Number(left & right))
                        }
                        (Value::Number(left), Value::Number(right), BinaryOperator::BitwiseOr) => {
                            Ok(Value::Number(left | right))
                        }
                        (Value::Number(left), Value::Number(right), BinaryOperator::BitwiseXor) => {
                            Ok(Value::Number(left ^ right))
                        }
                        (Value::Number(left), Value::Number(right), BinaryOperator::Divide) => {
                            if right == 0 {
                                Error::failure("Divide by 0")
                            } else {
                                Ok(Value::Number(left / right))
                            }
                        }
                        (Value::Number(left), Value::Number(right), BinaryOperator::Eq) => {
                            Ok(Value::Bool(left == right))
                        }
                        (Value::Number(left), Value::Number(right), BinaryOperator::Spaceship) => {
                            Ok(Value::Bool(left == right))
                        }
                        (Value::Number(left), Value::Number(right), BinaryOperator::NotEq) => {
                            Ok(Value::Bool(left != right))
                        }
                        (Value::Number(left), Value::Number(right), BinaryOperator::Gt) => {
                            Ok(Value::Bool(left > right))
                        }
                        (Value::Number(left), Value::Number(right), BinaryOperator::GtEq) => {
                            Ok(Value::Bool(left >= right))
                        }
                        (Value::Number(left), Value::Number(right), BinaryOperator::Lt) => {
                            Ok(Value::Bool(left < right))
                        }
                        (Value::Number(left), Value::Number(right), BinaryOperator::LtEq) => {
                            Ok(Value::Bool(left <= right))
                        }

                        (Value::Float(left), Value::Float(right), BinaryOperator::Plus) => {
                            Ok(Value::Float(left + right))
                        }
                        (Value::Float(left), Value::Float(right), BinaryOperator::Minus) => {
                            Ok(Value::Float(left - right))
                        }
                        (Value::Float(left), Value::Float(right), BinaryOperator::Multiply) => {
                            Ok(Value::Float(left * right))
                        }
                        (Value::Float(left), Value::Float(right), BinaryOperator::Modulus) => {
                            Ok(Value::Float(left % right))
                        }
                        (Value::Float(left), Value::Float(right), BinaryOperator::Divide) => {
                            if right == 0.0 {
                                Error::failure("Divide by 0")
                            } else {
                                Ok(Value::Float(left / right))
                            }
                        }
                        (Value::Float(left), Value::Float(right), BinaryOperator::Eq) => {
                            Ok(Value::Bool(left == right))
                        }
                        (Value::Float(left), Value::Float(right), BinaryOperator::Spaceship) => {
                            Ok(Value::Bool(left == right))
                        }
                        (Value::Float(left), Value::Float(right), BinaryOperator::NotEq) => {
                            Ok(Value::Bool(left != right))
                        }
                        (Value::Float(left), Value::Float(right), BinaryOperator::Gt) => {
                            Ok(Value::Bool(left > right))
                        }
                        (Value::Float(left), Value::Float(right), BinaryOperator::GtEq) => {
                            Ok(Value::Bool(left >= right))
                        }
                        (Value::Float(left), Value::Float(right), BinaryOperator::Lt) => {
                            Ok(Value::Bool(left < right))
                        }
                        (Value::Float(left), Value::Float(right), BinaryOperator::LtEq) => {
                            Ok(Value::Bool(left <= right))
                        }

                        (Value::String(left), Value::String(right), BinaryOperator::Eq) => {
                            Ok(Value::Bool(left == right))
                        }
                        (Value::String(left), Value::String(right), BinaryOperator::Spaceship) => {
                            Ok(Value::Bool(left == right))
                        }
                        (Value::String(left), Value::String(right), BinaryOperator::NotEq) => {
                            Ok(Value::Bool(left != right))
                        }
                        (Value::String(left), Value::String(right), BinaryOperator::Gt) => {
                            Ok(Value::Bool(left > right))
                        }
                        (Value::String(left), Value::String(right), BinaryOperator::GtEq) => {
                            Ok(Value::Bool(left >= right))
                        }
                        (Value::String(left), Value::String(right), BinaryOperator::Lt) => {
                            Ok(Value::Bool(left < right))
                        }
                        (Value::String(left), Value::String(right), BinaryOperator::LtEq) => {
                            Ok(Value::Bool(left <= right))
                        }
                        (Value::String(left), Value::String(right), BinaryOperator::Like) => {
                            Ok(Value::Bool(is_string_like(left.as_str(), right.as_str())))
                        }
                        (Value::String(left), Value::String(right), BinaryOperator::NotLike) => {
                            Ok(Value::Bool(!is_string_like(left.as_str(), right.as_str())))
                        }
                        (
                            Value::String(mut left),
                            Value::String(right),
                            BinaryOperator::StringConcat,
                        ) => {
                            left.push_str(right.as_str());

                            Ok(Value::String(left))
                        }

                        (Value::Bool(left), Value::Bool(right), BinaryOperator::And) => {
                            Ok(Value::Bool(left && right))
                        }
                        (Value::Bool(left), Value::Bool(right), BinaryOperator::Or) => {
                            Ok(Value::Bool(left || right))
                        }
                        (Value::Bool(left), Value::Bool(right), BinaryOperator::Eq) => {
                            Ok(Value::Bool(left == right))
                        }
                        (Value::Bool(left), Value::Bool(right), BinaryOperator::Spaceship) => {
                            Ok(Value::Bool(left == right))
                        }
                        (Value::Bool(left), Value::Bool(right), BinaryOperator::NotEq) => {
                            Ok(Value::Bool(left != right))
                        }
                        (Value::Bool(left), Value::Bool(right), BinaryOperator::Gt) => {
                            Ok(Value::Bool(left > right))
                        }
                        (Value::Bool(left), Value::Bool(right), BinaryOperator::GtEq) => {
                            Ok(Value::Bool(left >= right))
                        }
                        (Value::Bool(left), Value::Bool(right), BinaryOperator::Lt) => {
                            Ok(Value::Bool(left < right))
                        }
                        (Value::Bool(left), Value::Bool(right), BinaryOperator::LtEq) => {
                            Ok(Value::Bool(left <= right))
                        }

                        (left, right, op) => Error::failure(format!(
                            "Unsupported binary operation: {} {} {}",
                            left, op, right
                        )),
                    }?;

                    stack.push(Op::Value(result));
                }

                Op::Unary(op) => {
                    let expr = stack_pop(&mut params)?.as_value()?;

                    let result = match (expr, op) {
                        (Value::Number(value), UnaryOperator::Plus) => Ok(Value::Number(value)),
                        (Value::Number(value), UnaryOperator::Minus) => Ok(Value::Number(-value)),

                        (Value::Float(value), UnaryOperator::Plus) => Ok(Value::Float(value)),
                        (Value::Float(value), UnaryOperator::Minus) => Ok(Value::Float(-value)),

                        (Value::Bool(value), UnaryOperator::Not) => Ok(Value::Bool(!value)),

                        (value, op) => {
                            Error::failure(format!("Unsupported unary operation: {} {}", op, value))
                        }
                    }?;

                    stack.push(Op::Value(result));
                }

                Op::Value(value) => params.push(types::Param::Value(value)),

                Op::Between(negated) => {
                    let expr = stack_pop(&mut params)?.as_value()?;
                    let low = stack_pop(&mut params)?.as_value()?;
                    let high = stack_pop(&mut params)?.as_value()?;

                    let mut result = match (expr, low, high) {
                        (Value::Number(value), Value::Number(low), Value::Number(high)) => {
                            Ok(value >= low && value <= high)
                        }
                        (Value::Float(value), Value::Float(low), Value::Float(high)) => {
                            Ok(value >= low && value <= high)
                        }
                        (Value::String(value), Value::String(low), Value::String(high)) => {
                            Ok(value >= low && value <= high)
                        }
                        (expr, low, high) => {
                            let negated_str = if negated { "NOT" } else { "" };

                            Error::failure(format!(
                                "Invalid between arguments: {} {} BETWEEN {} AND {}",
                                expr, negated_str, low, high
                            ))
                        }
                    }?;

                    if negated {
                        result = !result;
                    }

                    stack.push(Op::Value(Value::Bool(result)));
                }

                Op::IsInList(negated) => {
                    let mut result = false;
                    let expr = stack_pop(&mut params)?.as_value()?;

                    while let Some(elem) = params.pop().map(|p| p.as_value()).transpose()? {
                        if !expr.is_same_type(&elem) {
                            return Error::failure("IN LIST operation contains elements that have a different time than target expression");
                        }

                        result = match (&expr, elem) {
                            (Value::Number(ref x), Value::Number(y)) => Ok(*x == y),
                            (Value::Float(ref x), Value::Float(y)) => {
                                // We all know doing equality checks over floats is stupid but it is what it is.
                                Ok(*x == y)
                            }
                            (Value::String(ref x), Value::String(ref y)) => Ok(x == y),
                            (Value::Bool(ref x), Value::Bool(y)) => Ok(*x == y),

                            _ => Error::failure(
                                "Supposedly unreachable code path reached: IN LIST evaluation",
                            ),
                        }?;

                        if result {
                            break;
                        }
                    }

                    if negated {
                        result = !result;
                    }

                    params.clear();
                    stack.push(Op::Value(Value::Bool(result)));
                }

                Op::IsNull(negated) => {
                    let mut result = if let Value::Null = stack_pop(&mut params)?.as_value()? {
                        true
                    } else {
                        false
                    };

                    if negated {
                        result = !result;
                    }

                    stack.push(Op::Value(Value::Bool(result)));
                }

                Op::InSubQuery {
                    subquery,
                    negated,
                    mut data,
                    current_line,
                } => {
                    let expr = stack_pop(&mut params)?.as_value()?;
                    let predicate_result = stack_pop(&mut params)?.as_value()?;

                    if expr.is_null() {
                        stack.push(Op::Value(expr));

                        continue;
                    }

                    let skip = predicate_result.is_null() || predicate_result.as_bool()?;

                    if !skip {
                        if current_line.len() == 1 {
                            for (_, elem) in current_line {
                                if !expr.is_same_type(&elem) {
                                    return Error::failure(format!("Different types used when consuming sub-query {:?} and {:?}", expr, elem));
                                }

                                match (&expr, elem) {
                                    (Value::Number(ref expr), Value::Number(elem))
                                        if expr == elem =>
                                    {
                                        stack.push(Op::Value(Value::Bool(!negated)));
                                    }

                                    (Value::Float(ref expr), Value::Float(elem))
                                        if expr == elem =>
                                    {
                                        stack.push(Op::Value(Value::Bool(!negated)));
                                    }

                                    (Value::String(ref expr), Value::String(ref elem))
                                        if expr == elem =>
                                    {
                                        stack.push(Op::Value(Value::Bool(!negated)));
                                    }

                                    (Value::Bool(ref expr), Value::Bool(elem)) if expr == elem => {
                                        stack.push(Op::Value(Value::Bool(!negated)));
                                    }

                                    (expr, elem) => {
                                        return Error::failure(format!("Unreachable code path reached in InSubQuery evaluation: {} {}", expr, elem));
                                    }
                                }
                            }

                            continue;
                        }

                        return Error::failure("in-sub query must only have one column");
                    }

                    if let Some(line) = data.next() {
                        let selection = main_info.sub_query_selection(subquery);
                        stack.push(Op::InSubQuery {
                            subquery,
                            negated,
                            data,
                            current_line: line,
                        });

                        // We push back onto the stack the left-side expression we already computed.
                        stack.push(Op::Value(expr));

                        if let Some(selection) = selection {
                            env.merge_scope(line.clone());
                            stack.push(Op::Return);
                            stack.push(Op::Expr(selection));
                        } else {
                            stack.push(Op::Value(Value::Bool(true)));
                        }

                        continue;
                    }

                    stack.push(Op::Value(Value::Bool(negated)));
                }

                Op::Expr(expr) => match expr {
                    Expr::Identifier(ident) => {
                        let value = env.resolve_name(&ident.value)?;

                        stack.push(Op::Value(value.clone()));
                    }

                    Expr::CompoundIdentifier(idents) => {
                        let name = types::flatten_idents(&idents);
                        let value = env.resolve_name(&name)?;

                        stack.push(Op::Value(value.clone()));
                    }

                    Expr::Value(value) => {
                        stack.push(Op::Value(Value::from_sql_value(value)?));
                    }

                    Expr::BinaryOp { left, op, right } => {
                        stack.push(Op::Binary(op.clone()));
                        stack.push(Op::Expr(left));
                        stack.push(Op::Expr(right));
                    }

                    Expr::UnaryOp { op, expr } => {
                        stack.push(Op::Unary(op.clone()));
                        stack.push(Op::Expr(expr));
                    }

                    Expr::IsNull(expr) => {
                        stack.push(Op::IsNull(false));
                        stack.push(Op::Expr(expr));
                    }

                    Expr::IsNotNull(expr) => {
                        stack.push(Op::IsNull(true));
                        stack.push(Op::Expr(expr));
                    }

                    Expr::Between {
                        expr,
                        negated,
                        low,
                        high,
                    } => {
                        stack.push(Op::Between(*negated));
                        stack.push(Op::Expr(expr));
                        stack.push(Op::Expr(low));
                        stack.push(Op::Expr(high));
                    }

                    Expr::InList {
                        expr,
                        list,
                        negated,
                    } => {
                        stack.push(Op::IsInList(*negated));
                        stack.push(Op::Expr(expr));

                        for elem in list {
                            stack.push(Op::Expr(elem));
                        }
                    }

                    Expr::Nested(expr) => {
                        stack.push(Op::Expr(expr));
                    }

                    Expr::InSubquery {
                        expr,
                        subquery,
                        negated,
                    } => {
                        let sub_query_info = main_info
                            .sub_query_info(subquery)
                            .expect("suq-query info must be defined");
                        let sub_query_name = &sub_query_info.name;

                        let mut data = sub_query_register
                            .get(sub_query_name)
                            .expect("sub-query data must be defined")
                            .iter();

                        if let Some(current_line) = data.next() {
                            stack.push(Op::InSubQuery {
                                subquery,
                                negated: *negated,
                                data,
                                current_line,
                            });
                            stack.push(Op::Expr(expr));

                            if let Some(selection) = main_info.sub_query_selection(subquery) {
                                stack.push(Op::Return);
                                stack.push(Op::Expr(selection));
                                env.merge_scope(current_line.clone());
                            } else {
                                stack.push(Op::Value(Value::Bool(true)));
                            }
                        } else {
                            stack.push(Op::InSubQuery {
                                subquery,
                                negated: *negated,
                                data,
                                current_line: &empty_line,
                            });

                            // No need to evaluate the left-side operand because
                            // the right side is NULL.
                            stack.push(Op::Value(Value::Null));
                            stack.push(Op::Value(Value::Null));
                        }
                    }

                    expr => return Error::failure(format!("Unsupported expression: {}", expr)),
                },
            }
        } else {
            result = params.pop().map(|p| p.as_value()).transpose()?;
            break;
        }
    }

    if let Some(result) = result {
        Ok(Either::Right(result))
    } else {
        Error::failure("Unexpected runtime error, no final result")
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Like {
    Pourcentage,
    Underscore,
    String(String),
}

fn parse_like_expr(expr: &str) -> Vec<Like> {
    let mut tokens = Vec::new();
    let mut buffer = String::new();
    let mut prev_was_pourcentage = false;
    let mut escaped = false;

    for c in expr.chars() {
        if c == '\\' {
            escaped = true;
            continue;
        }

        if !escaped && (c == '%' || c == '_') {
            if !buffer.is_empty() {
                tokens.push(Like::String(buffer));
                buffer = String::new();
            }

            if c == '%' && !prev_was_pourcentage {
                tokens.push(Like::Pourcentage);
                prev_was_pourcentage = true;
            } else if c == '_' {
                tokens.push(Like::Underscore);
                prev_was_pourcentage = false;
            }
        } else {
            buffer.push(c);
            prev_was_pourcentage = false;
            escaped = false;
        }
    }

    if !buffer.is_empty() {
        tokens.push(Like::String(buffer));
    }

    tokens
}

fn is_string_like(target: &str, expr: &str) -> bool {
    use Like::*;
    let instrs = parse_like_expr(expr);

    match instrs.as_slice() {
        [String(expr), Pourcentage] => target.starts_with(expr.as_str()),
        [Pourcentage, String(expr)] => target.ends_with(expr.as_str()),
        [Pourcentage, String(expr), Pourcentage] => target.contains(expr.as_str()),
        [String(start), Pourcentage, String(end)] => {
            target.starts_with(start.as_str()) && target.ends_with(end.as_str())
        }
        instrs => {
            let len = instrs.len();
            if len == 1 {
                return match &instrs[0] {
                    Underscore => target.len() == 1,
                    Pourcentage => true,
                    String(expr) => target == expr,
                };
            }

            let mut pattern = std::string::String::new();
            let mut offset = 0usize;

            for instr in instrs {
                match instr {
                    Pourcentage => {
                        if offset == 0 {
                            pattern.push('$');
                        } else if offset == len - 1 {
                            pattern.insert(0, '^');
                        }
                    }

                    Underscore => pattern.push('.'),
                    String(expr) => {
                        for c in expr.chars() {
                            if c.is_ascii_punctuation() {
                                pattern.push('\\');
                            }

                            pattern.push(c);
                        }
                    }
                }

                offset += 1;
            }

            // TODO - Cache compiled REGEX because those are expensive.
            let regex = regex::Regex::new(pattern.as_str()).unwrap();

            regex.is_match(target)
        }
    }
}

#[cfg(test)]
mod like_tests {
    use crate::types::{self, Line, line_insert};

    #[test]
    fn like_expr_parsing() {
        use super::parse_like_expr;
        use super::Like::*;

        assert_eq!(
            parse_like_expr("a%"),
            vec![String("a".to_string()), Pourcentage]
        );
        assert_eq!(
            parse_like_expr("%a"),
            vec![Pourcentage, String("a".to_string())]
        );
        assert_eq!(
            parse_like_expr("%or%"),
            vec![Pourcentage, String("or".to_string()), Pourcentage]
        );
        assert_eq!(
            parse_like_expr("_r%"),
            vec![Underscore, String("r".to_string()), Pourcentage]
        );
        assert_eq!(
            parse_like_expr("a_%"),
            vec![String("a".to_string()), Underscore, Pourcentage]
        );
        assert_eq!(
            parse_like_expr("a__%"),
            vec![String("a".to_string()), Underscore, Underscore, Pourcentage]
        );
        assert_eq!(
            parse_like_expr("a%o"),
            vec![
                String("a".to_string()),
                Pourcentage,
                String("o".to_string())
            ]
        );
        assert_eq!(parse_like_expr("%%"), vec![Pourcentage]);
        assert_eq!(parse_like_expr("%%%"), vec![Pourcentage]);
        assert_eq!(parse_like_expr("\\%"), vec![String("%".to_string())]);
    }

    #[test]
    fn like_predicate_test() {
        use super::is_string_like;

        assert!(is_string_like("ak", "a%"));
        assert!(is_string_like("ka", "%a"));
        assert!(is_string_like("bord", "%or%"));
        assert!(is_string_like("arddifskj", "_r%"));
        assert!(is_string_like("ax", "a_%"));
        assert!(is_string_like("axx", "a__%"));
        assert!(is_string_like("abazfoo", "a%o"));
    }

    struct Dummy;

    #[async_trait::async_trait]
    impl crate::types::DataSource for Dummy {
        async fn fetch(
            &self,
            _name: &crate::types::SourceName,
        ) -> crate::Result<futures::stream::BoxStream<'_, crate::Result<crate::types::Line>>>
        {
            todo!()
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn selection_test() -> crate::Result<()> {
        use futures::TryStreamExt;

        let query = "select 1 + 2 from toto join foo on x = y join bar on x = z";
        let query =
            sqlparser::parser::Parser::parse_sql(&sqlparser::dialect::AnsiDialect {}, query)
                .unwrap()
                .pop()
                .unwrap();

        let mut stream = super::run(Dummy, query).await?;

        while let Some(line) = stream.try_next().await? {
            println!(">>> {:?}", line);
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn jointure_test1() -> crate::Result<()> {
        use futures::{StreamExt, TryStreamExt};

        struct Source;

        #[async_trait::async_trait]
        impl crate::types::DataSource for Source {
            async fn fetch(
                &self,
                name: &crate::types::SourceName,
            ) -> crate::Result<futures::stream::BoxStream<'_, crate::Result<crate::types::Line>>>
            {
                if name.name == "foo" {
                    let lines = async_stream::try_stream! {
                        let mut line = crate::types::new_line();

                        line_insert(&mut line, "a", "a_string");
                        line_insert(&mut line, "c", 3);

                        yield line;

                        let mut line = crate::types::new_line();

                        line_insert(&mut line, "a", "xyz");
                        line_insert(&mut line, "c", 4);

                        yield line;
                    };

                    return Ok(lines.boxed());
                }

                if name.name == "bar" {
                    let lines = async_stream::try_stream! {
                        let mut line = crate::types::new_line();

                        line_insert(&mut line, "b", "monad");
                        line_insert(&mut line, "c", 7);

                        yield line;

                        let mut line = crate::types::new_line();

                        line_insert(&mut line, "b", "applicative");
                        line_insert(&mut line, "c", 3);

                        yield line;
                    };

                    return Ok(lines.boxed());
                }

                todo!()
            }
        }

        let query = "select foo.a, bar.b from foo as foo join bar as bar on bar.c = foo.c";
        let query =
            sqlparser::parser::Parser::parse_sql(&sqlparser::dialect::AnsiDialect {}, query)
                .unwrap()
                .pop()
                .unwrap();

        let mut result: Vec<Line> = super::run(Source, query).await?.try_collect().await?;

        assert!(!result.is_empty());

        let line = result.pop().unwrap();

        assert_eq!(line.len(), 2);
        assert_eq!(line.get(&"foo.a".to_string()).unwrap().as_str()?, "a_string");
        assert_eq!(line.get(&"bar.b".to_string()).unwrap().as_str()?, "applicative");

        Ok(())
    }
}
