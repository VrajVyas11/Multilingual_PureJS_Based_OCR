// types.d.ts
// Type definitions for multilingual-purejs-ocr
// Generated to provide TypeScript support for the pure JavaScript implementation.

import { InferenceSession, Tensor } from 'onnxruntime-node';

// =============================================================================
// Core Interfaces
// =============================================================================

export interface Box {
  left: number;
  top: number;
  width: number;
  height: number;
}

export type Polygon = number[][]; // Array of [x, y] points, e.g., [[x1, y1], [x2, y2], ...]

export interface TextElement {
  text: string;
  confidence: number;
  frame: Box;
  box: Polygon;
}

export interface Paragraph {
  text: string;
  confidence: number;
  boundingBox: Box;
  elements: TextElement[];
}

export interface OCRResult {
  totalElements: number;
  data: TextElement[];
  totalParagraphs?: number;
  paragraphs?: Paragraph[];
}

// =============================================================================
// Configuration Interfaces
// =============================================================================

export interface OnnxOptions {
  executionProviders?: string[];
  graphOptimizationLevel?: 'all' | 'basic' | 'extended' | 'off';
  enableCpuMemArena?: boolean;
  enableMemPattern?: boolean;
  executionMode?: 'sequential' | 'parallel';
  logSeverityLevel?: 0 | 1 | 2 | 3 | 4;
  intraOpNumThreads?: number;
  interOpNumThreads?: number;
}

export interface GroupingConfig {
  VERTICAL_THRESHOLD_RATIO: number;
  HORIZONTAL_THRESHOLD_RATIO: number;
  MIN_OVERLAP_RATIO: number;
  MAX_VERTICAL_OFFSET_RATIO: number;
}

export interface LanguageConfig {
  MODEL: string;
  DICT: string;
  NAME: string;
}

export type SupportedLanguage = 'en' | 'ch' | 'ja' | 'ko' | 'latin';

export interface RecognitionLanguages {
  en: LanguageConfig;
  ch: LanguageConfig;
  ja: LanguageConfig;
  ko: LanguageConfig;
  latin: LanguageConfig;
}

export interface DetectionConfig {
  MODEL_PATH: string;
  THRESHOLD: number;
  MIN_BOX_SIZE: number;
  MAX_BOX_SIZE: number;
  UNCLIP_RATIO: number;
  BASE_SIZE: number;
  MAX_IMAGE_SIZE: number;
  ONNX_OPTIONS: OnnxOptions;
}

export interface RecognitionConfig {
  LANGUAGES: RecognitionLanguages;
  DEFAULT_LANGUAGE: SupportedLanguage;
  IMAGE_HEIGHT: number;
  CONFIDENCE_THRESHOLD: number;
  REMOVE_DUPLICATE_CHARS: boolean;
  IGNORED_TOKENS: number[];
  ONNX_OPTIONS: OnnxOptions;
}

export interface DefaultConfig {
  DETECTION: DetectionConfig;
  RECOGNITION: RecognitionConfig;
  GROUPING: GroupingConfig;
}

export interface OcrOptions {
  // Language
  language?: SupportedLanguage;

  // Detection
  detectionThreshold?: number;
  minBoxSize?: number;
  maxBoxSize?: number;
  unclipRatio?: number;
  baseSize?: number;
  maxImageSize?: number;
  detectionModelPath?: string;

  // Recognition
  confidenceThreshold?: number;
  imageHeight?: number;
  removeDuplicateChars?: boolean;
  recognitionModelPath?: string;
  dictionaryPath?: string;

  // Grouping
  grouping?: Partial<GroupingConfig>;

  // ONNX
  detectionOnnxOptions?: Partial<OnnxOptions>;
  recognitionOnnxOptions?: Partial<OnnxOptions>;
}

export interface DetectOptions {
  grouped?: boolean;
  onnxOptions?: Partial<OnnxOptions>;
}

// =============================================================================
// Ocr Class Declaration (with Static Methods)
// =============================================================================

declare class Ocr {
  detect(imagePath: string, options?: DetectOptions): Promise<OCRResult>;

  setGroupingConfig(config: Partial<GroupingConfig>): void;

  getGroupingConfig(): GroupingConfig;

  static getAvailableLanguages(): RecognitionLanguages;

  static create(options?: OcrOptions): Promise<Ocr>;
}

export { Ocr };

// =============================================================================
// Internal Types (Exported for Completeness)
// =============================================================================

export interface DistanceResult {
  horizontal: number;
  vertical: number;
  euclidean: number;
}

export interface LineImage {
  box: Polygon;
  image: any; // ImageRaw instance (internal)
}

export interface ModelData {
  data: number[];
  width: number;
  height: number;
}

export interface DecodeResult {
  text: string;
  mean: number;
}

// Re-export for library users
export { InferenceSession, Tensor } from 'onnxruntime-node';

// Default config type
export const DEFAULT_CONFIG: DefaultConfig;

// Utility functions (typed for advanced users)
export function calculateDistance(box1: Box, box2: Box): DistanceResult;

export function groupTextElements(elements: TextElement[], config?: Partial<GroupingConfig>): Paragraph[][];

export function createParagraph(group: TextElement[]): Paragraph;