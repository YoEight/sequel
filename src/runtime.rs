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
                    }?;
                }

                Op::Value(value) => params.push(value),
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
