use crate::types::{self, Either, Error, Op, Suspension, Value};
use async_stream::stream;
use futures::stream::BoxStream;
use futures::TryStreamExt;
use sqlparser::ast::{BinaryOperator, Expr, UnaryOperator};

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
                let mut register = std::collections::HashMap::new();
                for name in info.source_name.iter() {
                    let stream = source.fetch(name).await?;
                    register.insert(name.clone(), stream);

                    for join_name in name.joins.iter() {
                        let stream = source.fetch(&join_name.source_name).await?;
                        register.insert(join_name.source_name.clone(), stream);
                    }
                }
                let mut env = types::Env::new();
                let mut offset = 0usize;

                if let sqlparser::ast::SetExpr::Select(select) = query.body {
                    if select.from.is_empty() {
                        let mut line = std::collections::HashMap::new();

                        for item in select.projection {
                            match item {
                                sqlparser::ast::SelectItem::UnnamedExpr(expr) => {
                                    let value = evaluate(&mut env, &expr)?.into_right().unwrap();
                                    line.insert(offset.to_string(), value);

                                    offset += 1;
                                }

                                sqlparser::ast::SelectItem::ExprWithAlias { expr, alias } => {
                                    let value = evaluate(&mut env, &expr)?.into_right().unwrap();
                                    line.insert(alias.value, value);
                                }

                                _ => {}
                            }
                        }

                        yield line;
                    } else {
                        if let Some(main_source) = info.source_name.as_ref() {
                            // It means we need to load the main source table entirely in memory.
                            if info.contains_right_join() {

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
    expr: &'a Expr,
) -> crate::Result<Either<Suspension<'a>, types::Value>> {
    let mut params = Vec::<types::Param>::new();
    let mut stack = Vec::<Op<'a>>::new();
    let mut result = None;

    stack.push(Op::Expr(expr));

    loop {
        if let Some(expr) = stack.pop() {
            match expr {
                Op::Return => {
                    if !env.exit_scope() {
                        break;
                    }
                }

                Op::EndOfStream => {
                    params.push(types::Param::EndOfStream);
                }

                Op::Suspend(id, info) => {
                    let suspension = Suspension {
                        id,
                        execution_stack: stack,
                        params,
                    };

                    return Ok(Either::Left(suspension));
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

                Op::InSubQuery(id, predicate_expr, negated) => {
                    let expr = stack_pop(&mut params)?.as_value()?;
                    let elem = stack_pop(&mut params)?.as_value()?;

                    if expr.is_null() {
                        stack.push(Op::Value(expr));
                        params.clear();

                        continue;
                    }

                    let skip = elem.is_null() || !elem.as_bool()?;

                    if !skip {
                        // if line.len() == 1 {
                        //     for (_, elem) in line {
                        //         if !expr.is_same_type(&elem) {
                        //             return Error::failure(format!("Different types used when consuming subquery {:?} and {:?}", expr, elem));
                        //         }
                        //
                        //         match (&expr, elem) {
                        //             (Value::Number(ref expr), Value::Number(elem))
                        //                 if *expr == elem =>
                        //             {
                        //                 stack.push(Op::Value(Value::Bool(!negated)));
                        //             }
                        //
                        //             (Value::Float(ref expr), Value::Float(elem))
                        //                 if *expr == elem =>
                        //             {
                        //                 stack.push(Op::Value(Value::Bool(!negated)));
                        //             }
                        //
                        //             (Value::String(ref expr), Value::String(ref elem))
                        //                 if expr == elem =>
                        //             {
                        //                 stack.push(Op::Value(Value::Bool(!negated)));
                        //             }
                        //
                        //             (Value::Bool(ref expr), Value::Bool(elem)) if *expr == elem => {
                        //                 stack.push(Op::Value(Value::Bool(!negated)));
                        //             }
                        //
                        //             (expr, elem) => {
                        //                 return Error::failure(format!("Unreachable code path reached in InSubQuery evaluation: {} {}", expr, elem));
                        //             }
                        //         }
                        //     }
                        //
                        //     continue;
                        // }

                        return Error::failure("in-sub query must only have one column");
                    }

                    // match stream.try_next().await {
                    //     Err(e) => {
                    //         return Error::failure(format!("Error when consuming subquery: {}", e));
                    //     }
                    //
                    //     Ok(line) => {
                    //         if let Some(line) = line {
                    //             stack.push(Op::InSubQuery(
                    //                 predicate_expr.clone(),
                    //                 negated,
                    //                 line.clone(),
                    //                 stream,
                    //             ));
                    //
                    //             // We push back onto the stack the left-side expression we already computed.
                    //             stack.push(Op::Value(expr));
                    //
                    //             if let Some(predicate_expr) = predicate_expr {
                    //                 env.merge_scope(line);
                    //                 stack.push(Op::Return);
                    //                 stack.push(Op::Expr(predicate_expr));
                    //             } else {
                    //                 stack.push(Op::Value(Value::Bool(true)));
                    //             }
                    //
                    //             continue;
                    //         }
                    //
                    //         stack.push(Op::Value(Value::Bool(negated)));
                    //     }
                    // }
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
                        let id = uuid::Uuid::new_v4();
                        let info = types::collect_query_info(&subquery)?;

                        stack.push(Op::InSubQuery(id, *negated, None));
                        stack.push(Op::Suspend(id, info));
                        stack.push(Op::Expr(expr));
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
}
