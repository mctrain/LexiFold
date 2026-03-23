// CRC32 implementation for LexiFold archive integrity

const CRC32_TABLE: number[] = buildCrc32Table();

function buildCrc32Table(): number[] {
  const table: number[] = new Array<number>(256);
  for (let i = 0; i < 256; i++) {
    let crc = i;
    for (let j = 0; j < 8; j++) {
      if ((crc & 1) !== 0) {
        crc = ((crc >>> 1) ^ 0xEDB88320) >>> 0;
      } else {
        crc = (crc >>> 1) >>> 0;
      }
    }
    table[i] = crc >>> 0;
  }
  return table;
}

export function crc32(data: Uint8Array, offset: number = 0, length: number = -1): number {
  const len = length < 0 ? data.length - offset : length;
  let crc = 0xFFFFFFFF;
  for (let i = offset; i < offset + len; i++) {
    const byte = data[i];
    crc = (CRC32_TABLE[(crc ^ byte) & 0xFF] ^ (crc >>> 8)) >>> 0;
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

export function crc32FromArray(data: number[], offset: number = 0, length: number = -1): number {
  const len = length < 0 ? data.length - offset : length;
  let crc = 0xFFFFFFFF;
  for (let i = offset; i < offset + len; i++) {
    const byte = data[i] & 0xFF;
    crc = (CRC32_TABLE[(crc ^ byte) & 0xFF] ^ (crc >>> 8)) >>> 0;
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}
