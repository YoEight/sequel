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

    pub fn from_sql_value(value: &sqlparser::ast::Value) -> crate::Result<Self> {
        match value {
            sqlparser::ast::Value::Number(ref num_str, _) => {
                if num_str.contains('.') {
                    match num_str.parse::<f64>() {
                        Ok(value) => Ok(Value::Float(value)),
                        Err(e) => {
                            Error::failure(format!("Invalid float number value format: {}", e))
                        }
                    }
                } else {
                    match num_str.parse::<i64>() {
                        Ok(value) => Ok(Value::Number(value)),
                        Err(e) => Error::failure(format!("Invalid number value format: {}", e)),
                    }
                }
            }

            sqlparser::ast::Value::SingleQuotedString(ref value) => {
                Ok(Value::String(value.clone()))
            }
            sqlparser::ast::Value::DoubleQuotedString(ref value) => {
                Ok(Value::String(value.clone()))
            }
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

pub enum Op<'a> {
    Binary(sqlparser::ast::BinaryOperator),
    Unary(sqlparser::ast::UnaryOperator),
    IsNull(bool),
    IsInList(bool),
    Between(bool),
    Value(Value),
    Expr(&'a sqlparser::ast::Expr),
    InSubQuery(
        Option<&'a sqlparser::ast::Expr>,
        bool,
        Option<Line>,
    ),
    // Select(String),
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

fn join_expr(constraint: &sqlparser::ast::JoinConstraint) -> Option<sqlparser::ast::Expr> {
    if let sqlparser::ast::JoinConstraint::On(expr) = constraint {
        Some(expr.clone())
    } else {
        None
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}

#[derive(Debug, Clone)]
pub struct Join {
    pub r#type: JoinType,
    pub source_name: SourceName,
    pub expr: Option<sqlparser::ast::Expr>,
}

#[derive(Debug, Clone)]
pub struct SourceName {
    alias: Option<String>,
    name: String,
    joins: Vec<Join>,
}

impl SourceName {
    pub fn alias(&self) -> Option<&String> {
        self.alias.as_ref()
    }

    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn joins(&self) -> &[Join] {
        self.joins.as_slice()
    }
}

pub struct QueryInfo {
    pub fields: Vec<String>,
    pub source_names: Vec<SourceName>,
    pub selection: Option<sqlparser::ast::Expr>,
}

pub fn collect_query_info(source: &sqlparser::ast::Query) -> crate::Result<QueryInfo> {
    let mut info = QueryInfo {
        fields: Vec::new(),
        source_names: Vec::new(),
        selection: None,
    };

    if let sqlparser::ast::SetExpr::Select(ref select) = source.body {
        for item in select.projection.iter() {
            if let sqlparser::ast::SelectItem::UnnamedExpr(expr) = item {
                match expr {
                    sqlparser::ast::Expr::Identifier(ident) => {
                        info.fields.push(ident.value.clone());
                    }

                    sqlparser::ast::Expr::CompoundIdentifier(idents) => {
                        info.fields.push(flatten_idents(&idents));
                    }

                    _ => {}
                }
            }
        }

        for from in select.from.iter() {
            if let sqlparser::ast::TableFactor::Table {
                ref name,
                ref alias,
                ..
            } = from.relation
            {
                let name = flatten_idents(&name.0);
                let alias = alias.as_ref().map(|a| a.name.value.clone());

                let mut joins = Vec::new();
                for join in from.joins.iter() {
                    if let sqlparser::ast::TableFactor::Table {
                        ref name,
                        ref alias,
                        ..
                    } = join.relation
                    {
                        let source_name = SourceName {
                            name: flatten_idents(&name.0),
                            alias: alias.as_ref().map(|a| a.name.value.clone()),
                            joins: Vec::new(),
                        };

                        let (r#type, expr) = match &join.join_operator {
                            sqlparser::ast::JoinOperator::Inner(ref expr) => {
                                (JoinType::Inner, join_expr(expr))
                            }
                            sqlparser::ast::JoinOperator::LeftOuter(ref expr) => {
                                (JoinType::Left, join_expr(expr))
                            }
                            sqlparser::ast::JoinOperator::RightOuter(ref expr) => {
                                (JoinType::Right, join_expr(expr))
                            }
                            sqlparser::ast::JoinOperator::FullOuter(ref expr) => {
                                (JoinType::Full, join_expr(expr))
                            }
                            unsupported => {
                                return Error::failure(format!(
                                    "Unsupported join strategy: {:?}",
                                    unsupported
                                ));
                            }
                        };

                        joins.push(Join {
                            r#type,
                            source_name,
                            expr,
                        });
                    } else {
                        return Error::failure(format!(
                            "Unsupported join naming strategy: {}",
                            join.relation
                        ));
                    }
                }

                info.source_names.push(SourceName { name, alias, joins });
            }
        }

        info.selection = select.selection.clone();
    }

    Ok(info)
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
        let parent = self
            .variables
            .get(&self.current_scope)
            .expect("current scope is always defined");
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

    pub fn resolve_name(&self, name: &String) -> crate::Result<&Value> {
        if let Some(value) = self
            .variables
            .get(&self.current_scope)
            .and_then(|line| line.get(name))
        {
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

#[async_trait::async_trait]
pub trait DataSource {
    async fn fetch(&self, name: &SourceName) -> crate::Result<BoxStream<'_, crate::Result<Line>>>;
}

#[derive(Debug, Clone)]
pub enum Param {
    Value(Value),
    Line(Line),
    EndOfStream,
}

impl Param {
    pub fn as_value(self) -> crate::Result<Value> {
        if let Param::Value(value) = self {
            Ok(value)
        } else {
            Error::failure(format!(
                "RUNTIME ERROR: Expected SQL Value but got: {:?}",
                self
            ))
        }
    }

    pub fn next_line(self) -> crate::Result<Option<Line>> {
        match self {
            Param::Line(line) => Ok(Some(line)),
            Param::EndOfStream => Ok(None),
            Param::Value(value) => Error::failure(format!(
                "RUNTIME ERROR: Expected stream but got: {:?}",
                value
            )),
        }
    }
}

pub enum SuspensionType {
    Next8,
}

pub struct Suspension<'a> {
    pub(crate) execution_stack: Vec<Op<'a>>,
    pub(crate) params: Vec<Value>,
}

#[derive(Debug, Eq, PartialEq)]
pub enum Either<A, B> {
    Left(A),
    Right(B),
}
