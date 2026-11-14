// The smallest job: one MI_STORE_DATA_IMM, written in the xe3 dialect directly (only xe3-layout runs on it).
image.memory @code attributes {mem = "streaming"} {
  image.stream @batch {
    xe3.cmd.mi_store_data_imm {address = @out, immediate_data = 0xacd0c0de}
  }
}
image.memory @result attributes {mem = "coherent"} {
  image.data @out {size = 8}
}
