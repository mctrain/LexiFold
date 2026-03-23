// PPM backend
export const ppmReset: () => void;
export const ppmGetFrequency: (symbol: number) => number[];
export const ppmGetTotal: () => number;
export const ppmFindSymbol: (scaledValue: number) => number[];
export const ppmUpdate: (symbol: number) => void;

// Neural GRU backend
export const nnLoadModel: (weightsBuffer: ArrayBuffer) => boolean;
export const nnReset: () => void;
export const nnGetFrequency: (symbol: number) => number[];
export const nnGetTotal: () => number;
export const nnFindSymbol: (scaledValue: number) => number[];
export const nnUpdate: (symbol: number) => void;
export const nnIsLoaded: () => boolean;
