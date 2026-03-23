export const ppmReset: () => void;
export const ppmGetFrequency: (symbol: number) => number[];
export const ppmGetTotal: () => number;
export const ppmFindSymbol: (scaledValue: number) => number[];
export const ppmUpdate: (symbol: number) => void;
