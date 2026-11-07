//! The machined stock. [`ImplicitStock`] is the independent checker: the stock a program
//! leaves, as an exact function of its moves. [`SweptStock`] is the in-process stock at
//! one slab that the generator's instruments measure engagement against; it is separate
//! code on purpose, so the checker never shares the generator's mistakes. [`StockTop`] is the
//! generator's stock top after a program, for the axial depth its plans read.

mod implicit;
mod swept;
mod top;

pub use implicit::{ImplicitStock, StockTables};
pub use swept::SweptStock;
pub use top::StockTop;
