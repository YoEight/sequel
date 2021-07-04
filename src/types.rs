use futures::stream::BoxStream;
use sqlparser::ast::{Expr, Query};
use std::collections::HashMap;
use std::slice::Iter;

#[derive(Debug, Clone)]
pub enum Value {
    Ident(String),
    Bool(bool),
    Number(i64),
    Float(f64),
    String(String),
    Null,
}

pub trait ToValue {
    fn to_value(&self) -> Value;
}

impl ToValue for String {
    fn to_value(&self) -> Value {
        Value::String(self.clone())
    }
}

impl ToValue for &str {
    fn to_value(&self) -> Value {
        Value::String(self.to_string())
    }
}

impl ToValue for bool {
    fn to_value(&self) -> Value {
        Value::Bool(*self)
    }
}

impl ToValue for i8 {
    fn to_value(&self) -> Value {
        Value::Number(*self as i64)
    }
}

impl ToValue for i16 {
    fn to_value(&self) -> Value {
        Value::Number(*self as i64)
    }
}

impl ToValue for i32 {
    fn to_value(&self) -> Value {
        Value::Number(*self as i64)
    }
}

impl ToValue for i64 {
    fn to_value(&self) -> Value {
        Value::Number(*self)
    }
}

impl ToValue for f32 {
    fn to_value(&self) -> Value {
        Value::Float(*self as f64)
    }
}

impl ToValue for f64 {
    fn to_value(&self) -> Value {
        Value::Float(*self)
    }
}

impl<A: ToValue> ToValue for Option<A> {
    fn to_value(&self) -> Value {
        match self.as_ref() {
            None => Value::Null,
            Some(a) => a.to_value(),
        }
    }
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

    pub fn as_str(&self) -> crate::Result<&str> {
        if let Value::String(value) = self {
            Ok(value.as_str())
        } else {
            Error::failure(format!("Expected String got: {}", self))
        }
    }

    pub fn as_num(&self) -> crate::Result<i64> {
        if let Value::Number(value) = self {
            Ok(*value)
        } else {
            Error::failure(format!("Expected number got: {}", self))
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
    InSubQuery {
        subquery: &'a Query,
        negated: bool,
        data: Iter<'a, Line>,
        current_line: &'a Line,
    },
    Return,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Scope(usize);

pub type Line = HashMap<String, Value>;
pub type Cache = HashMap<String, Vec<Line>>;
pub type Fields = Vec<String>;

pub fn new_line() -> Line {
    HashMap::new()
}

pub fn line_insert<A: ToValue>(line: &mut Line, key: impl AsRef<str>, value: A) {
    line.insert(key.as_ref().to_string(), value.to_value());
}

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

pub fn rename_line(source_name: &SourceName, mut line: Line) -> Line {
    if let Some(alias) = source_name.alias() {
        let mut renamed_line = HashMap::with_capacity(line.capacity());

        for (key, value) in line {
            renamed_line.insert(format!("{}.{}", alias, key), value);
        }

        line = renamed_line;
    }

    line
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

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Join {
    pub r#type: JoinType,
    pub source_name: SourceName,
    pub expr: Option<sqlparser::ast::Expr>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SourceName {
    pub alias: Option<String>,
    pub name: String,
    pub joins: Vec<Join>,
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

#[derive(Debug)]
pub struct SubQueryInfo {
    pub name: SourceName,
    pub selection: Option<Expr>,
}

#[derive(Debug)]
pub struct QueryInfo {
    pub fields: Vec<String>,
    pub source_name: Option<SourceName>,
    pub selection: Option<sqlparser::ast::Expr>,
    pub sub_queries_info: HashMap<Query, SubQueryInfo>,
}

impl QueryInfo {
    pub fn new() -> Self {
        Self {
            fields: Vec::new(),
            source_name: None,
            selection: None,
            sub_queries_info: HashMap::new(),
        }
    }

    pub fn contains_right_join(&self) -> bool {
        if let Some(source_name) = self.source_name.as_ref() {
            for join in source_name.joins.iter() {
                if join.r#type == JoinType::Right || join.r#type == JoinType::Full {
                    return true;
                }
            }
        }

        false
    }

    pub fn sub_queries(&self) -> impl Iterator<Item = &SourceName> {
        self.sub_queries_info.values().map(|info| &info.name)
    }

    pub fn sub_query_selection(&self, query: &Query) -> Option<&Expr> {
        self.sub_query_info(query)
            .and_then(|info| info.selection.as_ref())
    }

    pub fn sub_query_info(&self, query: &Query) -> Option<&SubQueryInfo> {
        self.sub_queries_info.get(query)
    }
}

pub fn collect_query_info(source: &sqlparser::ast::Query) -> crate::Result<QueryInfo> {
    let mut info = QueryInfo::new();

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

        // We only support single FROM list.
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

                info.source_name = Some(SourceName { name, alias, joins });
                break;
            }
        }

        info.selection = select.selection.clone();

        // We collect sub-queries to prefetch and ease their execution later on.
        if let Some(expr) = info.selection.as_ref() {
            let mut stack = vec![expr];
            while let Some(expr) = stack.pop() {
                match expr {
                    Expr::BinaryOp { left, right, .. } => {
                        stack.push(left);
                        stack.push(right);
                    }

                    Expr::UnaryOp { expr, .. } => {
                        stack.push(expr);
                    }

                    Expr::IsNull(expr) => {
                        stack.push(expr);
                    }

                    Expr::IsNotNull(expr) => {
                        stack.push(expr);
                    }

                    Expr::Nested(expr) => {
                        stack.push(expr);
                    }

                    Expr::InSubquery { subquery, .. } => {
                        if let sqlparser::ast::SetExpr::Select(ref select) = subquery.body {
                            for table in select.from.iter() {
                                if let sqlparser::ast::TableFactor::Table {
                                    ref name,
                                    ref alias,
                                    ..
                                } = table.relation
                                {
                                    let name = SourceName {
                                        name: flatten_idents(&name.0),
                                        alias: alias.as_ref().map(|a| a.name.value.clone()),
                                        joins: Vec::new(),
                                    };

                                    let selection = if let Some(expr) = select.selection.as_ref() {
                                        stack.push(expr);

                                        Some(expr.clone())
                                    } else {
                                        None
                                    };

                                    info.sub_queries_info.insert(
                                        *subquery.clone(),
                                        SubQueryInfo { name, selection },
                                    );
                                }
                            }
                        }
                    }

                    _ => {
                        // We don't do anything because in those situation,
                        // there is no way we will find a sub-query.
                    }
                }
            }
        }
    }

    Ok(info)
}

#[derive(Debug)]
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
        self.prev_scopes.push(self.current_scope);
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

    pub fn project_line(&self, info: &QueryInfo) -> crate::Result<Line> {

        if let Some(mut line) = self.variables.get(&self.current_scope).cloned() {
            project_line(&info.fields, &mut line);

            return Ok(line)

        }

        Error::failure(format!("Internal error: undefined current scope!"))
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
    pub(crate) id: uuid::Uuid,
    pub(crate) execution_stack: Vec<Op<'a>>,
    pub(crate) params: Vec<Param>,
}

#[derive(Debug, Eq, PartialEq)]
pub enum Either<A, B> {
    Left(A),
    Right(B),
}

impl<A, B> Either<A, B> {
    pub fn into_right(self) -> Option<B> {
        match self {
            Either::Right(b) => Some(b),
            _ => None,
        }
    }

    pub fn into_left(self) -> Option<A> {
        match self {
            Either::Left(a) => Some(a),
            _ => None,
        }
    }
}

pub type Register = HashMap<SourceName, Vec<Line>>;
