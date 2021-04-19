use futures::stream::BoxStream;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Value {
    Ident(String),
    Bool(bool),
    Number(i64),
    Float(f64),
    String(String),
    Null,
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Value::Ident(ref value) => write!(f, "{}", value),
            Value::Bool(ref value) => write!(f, "{}", value),
            Value::Number(ref value) => write!(f, "{}", value),
            Value::Float(ref value) => write!(f, "{}", value),
            Value::String(ref value) => write!(f, "{}", value),
            Value::Null => write!(f, "NULL"),
        }
    }
}

impl Value {
    pub fn is_same_type(&self, value: Value) -> bool {
        match (self, value) {
            (Value::Bool(_), Value::Bool(_)) => true,
            (Value::Number(_), Value::Number(_)) => true,
            (Value::Float(_), Value::Float(_)) => true,
            (Value::String(_), Value::String(_)) => true,
            _ => false,
        }
    }

    pub fn is_null(&self) -> bool {
        match self {
            Value::Null => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Error(String);

impl Error {
    pub fn failure<A, V: AsRef<str>>(value: V) -> crate::Result<A> {
        Err(Error(value.as_ref().to_string()))
    }
}

pub type Result<A> = std::result::Result<A, Error>;

pub enum Op {
    Binary(sqlparser::ast::BinaryOperator),
    Unary(sqlparser::ast::UnaryOperator),
    IsNull(bool),
    IsInList(bool),
    Between(bool),
    Value(Value),
    Expr(sqlparser::ast::Expr),
    InSsubQuery(bool, BoxStream<'static, Result<Line>>),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Scope(usize);

pub type Line = HashMap<String, Value>;
pub type Cache = HashMap<String, Vec<Line>>;

pub struct Env {
    scope_gen: usize,
    variables: HashMap<Scope, Line>,
}

impl Env {
    pub fn new() -> Self {
        Self {
            scope_gen: 0,
            variables: HashMap::new(),
        }
    }

    pub fn create_scope(&mut self, line: Line) -> Scope {
        let scope = Scope(self.scope_gen);

        self.scope_gen += 1;

        self.variables.insert(scope, line);

        scope
    }

    pub fn delete_scope(&mut self, scope: Scope) {
        self.variables.remove(&scope);
    }

    pub fn get_line(&self, scope: Scope) -> &Line {
        self.variables.get(&scope).expect("scope to be defined")
    }
}
