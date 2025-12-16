file(READ "${INPUT}" BINARY_DATA HEX)

file(WRITE "${OUTPUT}" "// Auto-generated\n")
file(APPEND "${OUTPUT}" "static const unsigned char ${SYMBOL}[] = {\n")

string(LENGTH "${BINARY_DATA}" LEN)
math(EXPR END "${LEN} - 2")

foreach(I RANGE 0 ${END} 2)
  string(SUBSTRING "${BINARY_DATA}" ${I} 2 BYTE)
  file(APPEND "${OUTPUT}" "0x${BYTE},")
endforeach()

file(APPEND "${OUTPUT}" "\n};\n")
