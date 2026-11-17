use mlirformat::Error;

use crate::{Body, OpId, pass};
use crate::{Xe3Batch};

#[pass("xe3-end", on = "builtin.module")]
pub fn xe3_end<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    Xe3Batch::end(body, op);
    Ok(())
}
