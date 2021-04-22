use crate::types::{self, Error, Op, Value};
use sqlparser::ast::{BinaryOperator, Expr, UnaryOperator};

fn stack_pop<A>(stack: &mut Vec<A>) -> crate::Result<A> {
    match stack.pop() {
        Some(op) => Ok(op),
        None => Error::failure("Unexpected empty stack"),
    }
}

pub fn evaluate(env: &mut types::Env, expr: Expr) -> crate::Result<types::Value> {
    let mut params = Vec::<Value>::new();
    let mut stack = Vec::<Op>::new();
    let mut result = None;

    stack.push(Op::Expr(expr));

    loop {
        if let Some(expr) = stack.pop() {
            match expr {
                Op::Binary(op) => {
                    let left = stack_pop(&mut params)?;
                    let right = stack_pop(&mut params)?;

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
                    let expr = stack_pop(&mut params)?;

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

                Op::Value(value) => params.push(value),

                Op::Between(negated) => {
                    let expr = stack_pop(&mut params)?;
                    let low = stack_pop(&mut params)?;
                    let high = stack_pop(&mut params)?;

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
                    let expr = stack_pop(&mut params)?;

                    while let Some(elem) = params.pop() {
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
                    let mut result = if let Value::Null = stack_pop(&mut params)? {
                        true
                    } else {
                        false
                    };

                    if negated {
                        result = !result;
                    }

                    stack.push(Op::Value(Value::Bool(result)));
                }

                _ => {}
            }
        } else {
            result = params.pop();
            break;
        }
    }

    if let Some(result) = result {
        Ok(result)
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
}
