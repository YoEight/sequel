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

                        _ => { Error::failure("toto") }
                    }?;
                }

                Op::Value(value) => params.push(value),
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

    for c in expr.chars() {
        if c == '%' || c == '_' {
            if !buffer.is_empty() {
                tokens.push(Like::String(buffer));
                buffer = String::new();
            }

            if c == '%' {
                tokens.push(Like::Pourcentage);
            } else if c == '_' {
                tokens.push(Like::Underscore);
            }
        } else {
            buffer.push(c);
        }
    }

    if !buffer.is_empty() {
        tokens.push(Like::String(buffer));
    }

    tokens
}

#[cfg(test)]
mod like_tests {

    #[test]
    fn like_parsing_1() {
        use super::Like::*;
        use super::parse_like_expr;

        assert_eq!(parse_like_expr("a%"), vec![String("a".to_string()), Pourcentage]);
        assert_eq!(parse_like_expr("%a"), vec![Pourcentage, String("a".to_string())]);
        assert_eq!(parse_like_expr("%or%"), vec![Pourcentage, String("or".to_string()), Pourcentage]);
        assert_eq!(parse_like_expr("_r%"), vec![Underscore, String("r".to_string()), Pourcentage]);
        assert_eq!(parse_like_expr("a_%"), vec![String("a".to_string()), Underscore, Pourcentage]);
        assert_eq!(parse_like_expr("a__%"), vec![String("a".to_string()), Underscore, Underscore, Pourcentage]);
        assert_eq!(parse_like_expr("a%o"), vec![String("a".to_string()), Pourcentage, String("o".to_string())]);
    }
}

fn is_string_like(target: &String, expr: String) -> bool {
    false
}