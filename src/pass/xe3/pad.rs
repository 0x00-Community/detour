use mlirformat::Error;

use crate::{Body, OpId, pass};
use crate::{Xe3Batch};

#[pass("xe3-pad", on = "builtin.module")]
pub fn xe3_pad<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    Xe3Batch::pad(body, op);
    Ok(())
}
