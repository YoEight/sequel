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
    pub fn is_same_type(&self, value: &Value) -> bool {
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

    pub fn as_bool(&self) -> crate::Result<bool> {
        if let Value::Bool(value) = self {
            Ok(*value)
        } else {
            Error::failure(format!("Expected boolean got: {}", self))
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
    InSubQuery(bool, BoxStream<'static, Result<Line>>),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Scope(usize);

pub type Line = HashMap<String, Value>;
pub type Cache = HashMap<String, Vec<Line>>;
pub type Fields = Vec<String>;

pub fn project_line(fields: &Fields, line: &mut Line) {
    if fields.is_empty() {
        return;
    }

    line.retain(|key, _| fields.contains(key));
}

pub fn collect_fields(select: &sqlparser::ast::Select) -> Fields {
    let mut fields = Vec::new();

    for item in select.projection.iter() {
        if let sqlparser::ast::SelectItem::UnnamedExpr(expr) = item {
            match expr {
                sqlparser::ast::Expr::Identifier(ident) => {
                    fields.push(ident.value.clone());
                }

                sqlparser::ast::Expr::CompoundIdentifier(idents) => {
                    let mut ident = String::new();

                    for elem in idents.iter() {
                        if ident.is_empty() {
                            ident.push_str(elem.value.as_str());
                        } else {
                            ident.push('.');
                            ident.push_str(elem.value.as_str());
                        }
                    }

                    fields.push(ident);
                }

                _ => {}
            }
        }
    }

    fields
}

pub struct Env {
    scope_gen: usize,
    variables: HashMap<Scope, Line>,
    query_fields: HashMap<sqlparser::ast::Expr, Fields>,
}

impl Env {
    pub fn new() -> Self {
        Self {
            scope_gen: 0,
            variables: HashMap::new(),
            query_fields: HashMap::new(),
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

    pub fn get_query_fields(&self, query: &sqlparser::ast::Expr) -> crate::Result<&Fields> {
        if let Some(fields) = self.query_fields.get(query) {
            Ok(fields)
        } else {
            Error::failure(format!("No field registered for query: {}", query))
        }
    }
}
