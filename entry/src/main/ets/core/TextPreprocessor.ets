// Text Preprocessor for LexiFold
// v0: identity preprocessor (no transformation)
// v1: reversible lowercase normalization

import { Preprocessor, PREPROCESSOR_IDENTITY, PREPROCESSOR_LOWERCASE } from './Types';

const ESCAPE_CHAR: string = '\u0001'; // SOH, almost never in natural text

// === Identity Preprocessor (v0) ===

export class IdentityPreprocessor implements Preprocessor {
  readonly id: number = PREPROCESSOR_IDENTITY;
  readonly name: string = 'None';

  preprocess(text: string): string {
    return text;
  }

  reverse(text: string): string {
    return text;
  }
}

// === Lowercase Preprocessor (v1) ===
// Reversible case normalization: uppercase → ESCAPE + lowercase.
// Reduces effective alphabet for better prediction on natural language.
// Encoding: 'A' → '\x01a', '\x01' → '\x01\x01', other → as-is.

export class LowercasePreprocessor implements Preprocessor {
  readonly id: number = PREPROCESSOR_LOWERCASE;
  readonly name: string = 'Lowercase';

  preprocess(text: string): string {
    let result: string = '';
    for (let i = 0; i < text.length; i++) {
      const ch: string = text.charAt(i);
      if (ch === ESCAPE_CHAR) {
        // Escape the escape character itself
        result += ESCAPE_CHAR + ESCAPE_CHAR;
      } else if (ch >= 'A' && ch <= 'Z') {
        // Uppercase ASCII → escape + lowercase
        result += ESCAPE_CHAR + ch.toLowerCase();
      } else {
        result += ch;
      }
    }
    return result;
  }

  reverse(text: string): string {
    let result: string = '';
    let i: number = 0;
    while (i < text.length) {
      const ch: string = text.charAt(i);
      if (ch === ESCAPE_CHAR && i + 1 < text.length) {
        const next: string = text.charAt(i + 1);
        if (next === ESCAPE_CHAR) {
          // Escaped escape → literal escape char
          result += ESCAPE_CHAR;
        } else {
          // Escaped lowercase → uppercase
          result += next.toUpperCase();
        }
        i += 2;
      } else {
        result += ch;
        i++;
      }
    }
    return result;
  }
}

// === Preprocessor Factory ===

export function createPreprocessor(preprocessorId: number): Preprocessor {
  if (preprocessorId === PREPROCESSOR_LOWERCASE) {
    return new LowercasePreprocessor();
  }
  // Default: identity
  return new IdentityPreprocessor();
}
