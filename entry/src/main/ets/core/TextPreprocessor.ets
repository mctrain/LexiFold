// Text Preprocessor for LexiFold
// v0: identity preprocessor (no transformation)
// v1: reversible text preprocessing (case normalization, etc.)

import { PREPROCESSOR_IDENTITY } from './Types';

export class IdentityPreprocessor {
  readonly id: number = PREPROCESSOR_IDENTITY;
  readonly name: string = 'Identity (None)';

  preprocess(text: string): string {
    return text;
  }

  reverse(text: string): string {
    return text;
  }
}
