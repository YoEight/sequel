#[macro_use]
extern crate log;

mod runtime;
mod types;

pub use types::Result;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
