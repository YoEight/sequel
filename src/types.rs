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

    pub fn from_sql_value(value: sqlparser::ast::Value) -> crate::Result<Self> {
        match value {
            sqlparser::ast::Value::Number(ref num_str, _) => {
                if num_str.contains('.') {
                    match num_str.parse::<f64>() {
                        Ok(value) => Ok(Value::Float(value)),
                        Err(e) => Error::failure(format!("Invalid float number value format: {}", e)),
                    }
                } else {
                    match num_str.parse::<i64>() {
                        Ok(value) => Ok(Value::Number(value)),
                        Err(e) => Error::failure(format!("Invalid number value format: {}", e)),
                    }
                }
            }

            sqlparser::ast::Value::SingleQuotedString(ref value) => Ok(Value::String(value.clone())),
            sqlparser::ast::Value::DoubleQuotedString(ref value) => Ok(Value::String(value.clone())),
            sqlparser::ast::Value::Boolean(ref value) => Ok(Value::Bool(*value)),
            sqlparser::ast::Value::Null => Ok(Value::Null),
            wrong => Error::failure(format!("Expected SQL value but got: {}", wrong)),
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

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
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
    InSubQuery(
        Option<sqlparser::ast::Expr>,
        bool,
        Line,
        BoxStream<'static, Result<Line>>,
    ),
    Return,
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
                    fields.push(flatten_idents(idents));
                }

                _ => {}
            }
        }
    }

    fields
}

pub fn flatten_idents(idents: &Vec<sqlparser::ast::Ident>) -> String {
    let mut ident = String::new();

    for elem in idents.iter() {
        if ident.is_empty() {
            ident.push_str(elem.value.as_str());
        } else {
            ident.push('.');
            ident.push_str(elem.value.as_str());
        }
    }

    ident
}

pub struct Env {
    scope_gen: usize,
    prev_scopes: Vec<Scope>,
    current_scope: Scope,
    variables: HashMap<Scope, Line>,
    query_fields: HashMap<sqlparser::ast::Expr, Fields>,
}

impl Env {
    pub fn new() -> Self {
        Self {
            scope_gen: 1,
            prev_scopes: Vec::new(),
            current_scope: Scope(0),
            variables: HashMap::new(),
            query_fields: HashMap::new(),
        }
    }

    pub fn enter_scope(&mut self, line: Line) {
        let scope = Scope(self.scope_gen);

        self.scope_gen += 1;
        self.current_scope = scope;
        self.variables.insert(scope, line);
    }

    pub fn merge_scope(&mut self, line: Line) {
        let parent = self.variables.get(&self.current_scope).expect("current scope is always defined");
        let mut parent = parent.clone();

        parent.extend(line);

        self.enter_scope(parent)
    }

    pub fn exit_scope(&mut self) -> bool {
        self.variables.remove(&self.current_scope);

        if let Some(prev_scope) = self.prev_scopes.pop() {
            self.current_scope = prev_scope;
            true
        } else {
            false
        }
    }

    // pub fn get_scope_line(&self) ->

    pub fn resolve_name(&self, name: &String) -> crate::Result<&Value> {
        if let Some(value) = self.variables.get(&self.current_scope).and_then(|line| line.get(name)) {
            Ok(value)
        } else {
            Error::failure(format!("Field {} not in scope", name))
        }
    }

    pub fn get_query_fields(&self, query: &sqlparser::ast::Expr) -> crate::Result<&Fields> {
        if let Some(fields) = self.query_fields.get(query) {
            Ok(fields)
        } else {
            Error::failure(format!("No field registered for query: {}", query))
        }
    }
}
