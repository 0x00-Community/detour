use mlirformat::Error;

use mlirpass::{Body, OpId, pass};
use super::{Xe3Image};

#[pass("xe3-end", on = "builtin.module")]
pub fn xe3_end<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    Xe3Image::end(body, op);
    Ok(())
}
