// Archive format read/write for LexiFold (.lxf files)
// Header: 26 bytes, followed by chunk data

import {
  ARCHIVE_MAGIC, ARCHIVE_VERSION, ARCHIVE_HEADER_SIZE, CHUNK_HEADER_SIZE,
  ArchiveHeader, ChunkMeta, ChunkWithData
} from './Types';
import { crc32 } from './Checksum';

// === Binary helpers ===

function writeUint8(buf: Uint8Array, offset: number, value: number): void {
  buf[offset] = value & 0xFF;
}

function writeUint16LE(buf: Uint8Array, offset: number, value: number): void {
  buf[offset] = value & 0xFF;
  buf[offset + 1] = (value >> 8) & 0xFF;
}

function writeUint32LE(buf: Uint8Array, offset: number, value: number): void {
  buf[offset] = value & 0xFF;
  buf[offset + 1] = (value >> 8) & 0xFF;
  buf[offset + 2] = (value >> 16) & 0xFF;
  buf[offset + 3] = (value >>> 24) & 0xFF;
}

function readUint8(buf: Uint8Array, offset: number): number {
  return buf[offset];
}

function readUint16LE(buf: Uint8Array, offset: number): number {
  return buf[offset] | (buf[offset + 1] << 8);
}

function readUint32LE(buf: Uint8Array, offset: number): number {
  return (buf[offset] | (buf[offset + 1] << 8) | (buf[offset + 2] << 16) | (buf[offset + 3] << 24)) >>> 0;
}

// === Header Serialization ===

export function writeArchiveHeader(header: ArchiveHeader): Uint8Array {
  const buf = new Uint8Array(ARCHIVE_HEADER_SIZE);

  // Magic "LXFD"
  buf[0] = ARCHIVE_MAGIC.charCodeAt(0);
  buf[1] = ARCHIVE_MAGIC.charCodeAt(1);
  buf[2] = ARCHIVE_MAGIC.charCodeAt(2);
  buf[3] = ARCHIVE_MAGIC.charCodeAt(3);

  writeUint8(buf, 4, header.version);
  writeUint8(buf, 5, header.flags);
  writeUint8(buf, 6, header.tokenizerId);
  writeUint8(buf, 7, header.preprocessorId);
  writeUint16LE(buf, 8, header.backendId);
  writeUint16LE(buf, 10, header.modelId);
  writeUint32LE(buf, 12, header.chunkSize);
  writeUint32LE(buf, 16, header.originalSize);
  writeUint16LE(buf, 20, header.numChunks);

  // Header checksum: CRC32 of bytes 0-21
  const checksum = crc32(buf, 0, 22);
  writeUint32LE(buf, 22, checksum);

  return buf;
}

export function readArchiveHeader(buf: Uint8Array): ArchiveHeader | null {
  if (buf.length < ARCHIVE_HEADER_SIZE) {
    return null;
  }

  // Check magic
  const magic = String.fromCharCode(buf[0], buf[1], buf[2], buf[3]);
  if (magic !== ARCHIVE_MAGIC) {
    return null;
  }

  const version = readUint8(buf, 4);
  if (version !== ARCHIVE_VERSION) {
    return null;
  }

  // Verify header checksum
  const storedChecksum = readUint32LE(buf, 22);
  const computedChecksum = crc32(buf, 0, 22);
  if (storedChecksum !== computedChecksum) {
    return null;
  }

  return {
    magic: magic,
    version: version,
    flags: readUint8(buf, 5),
    tokenizerId: readUint8(buf, 6),
    preprocessorId: readUint8(buf, 7),
    backendId: readUint16LE(buf, 8),
    modelId: readUint16LE(buf, 10),
    chunkSize: readUint32LE(buf, 12),
    originalSize: readUint32LE(buf, 16),
    numChunks: readUint16LE(buf, 20),
    headerChecksum: storedChecksum
  };
}

// === Chunk Header Serialization ===

export function writeChunkHeader(meta: ChunkMeta): Uint8Array {
  const buf = new Uint8Array(CHUNK_HEADER_SIZE);
  writeUint32LE(buf, 0, meta.compressedSize);
  writeUint32LE(buf, 4, meta.originalSize);
  writeUint32LE(buf, 8, meta.checksum);
  return buf;
}

export function readChunkHeader(buf: Uint8Array, offset: number): ChunkMeta | null {
  if (offset + CHUNK_HEADER_SIZE > buf.length) {
    return null;
  }
  return {
    compressedSize: readUint32LE(buf, offset),
    originalSize: readUint32LE(buf, offset + 4),
    checksum: readUint32LE(buf, offset + 8)
  };
}

// === Full Archive Assembly ===

export function assembleArchive(
  header: ArchiveHeader,
  chunks: ChunkWithData[]
): Uint8Array {
  // Calculate total size
  let totalSize = ARCHIVE_HEADER_SIZE;
  for (const chunk of chunks) {
    totalSize += CHUNK_HEADER_SIZE + chunk.data.length;
  }

  const archive = new Uint8Array(totalSize);

  // Write header
  const headerBytes = writeArchiveHeader(header);
  archive.set(headerBytes, 0);

  // Write chunks
  let offset = ARCHIVE_HEADER_SIZE;
  for (const chunk of chunks) {
    const chunkHeader = writeChunkHeader(chunk.meta);
    archive.set(chunkHeader, offset);
    offset += CHUNK_HEADER_SIZE;
    archive.set(chunk.data, offset);
    offset += chunk.data.length;
  }

  return archive;
}

// === Archive Parsing ===

export interface ParsedArchive {
  header: ArchiveHeader;
  chunks: ChunkWithData[];
}

export function parseArchive(buf: Uint8Array): ParsedArchive | null {
  const header = readArchiveHeader(buf);
  if (header === null) {
    return null;
  }

  const chunks: ChunkWithData[] = [];
  let offset = ARCHIVE_HEADER_SIZE;

  for (let i = 0; i < header.numChunks; i++) {
    const meta = readChunkHeader(buf, offset);
    if (meta === null) {
      return null;
    }
    offset += CHUNK_HEADER_SIZE;

    if (offset + meta.compressedSize > buf.length) {
      return null;
    }

    const data = buf.slice(offset, offset + meta.compressedSize);
    const chunk: ChunkWithData = { meta: meta, data: data };
    chunks.push(chunk);
    offset += meta.compressedSize;
  }

  const result: ParsedArchive = { header: header, chunks: chunks };
  return result;
}
