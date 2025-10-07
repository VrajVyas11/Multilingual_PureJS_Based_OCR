// index.js
import fs from 'node:fs/promises';
import invariant from 'tiny-invariant';
import { InferenceSession, Tensor } from 'onnxruntime-node';
import sharp from 'sharp';
import cv from '@techstark/opencv-js';
import clipper from 'js-clipper';

// =============================================================================
// DEFAULT CONFIGURATION
// =============================================================================

const DEFAULT_CONFIG = {
    DETECTION: {
        MODEL_PATH: './models/ch_PP-OCRv4_det_infer.onnx',
        THRESHOLD: 0.1,
        MIN_BOX_SIZE: 3,
        MAX_BOX_SIZE: 2000,
        UNCLIP_RATIO: 1.5,
        BASE_SIZE: 32,
        MAX_IMAGE_SIZE: 960,
        ONNX_OPTIONS: {
            executionProviders: ['cpu'],
            graphOptimizationLevel: 'all',
            enableCpuMemArena: true,
            enableMemPattern: true,
            executionMode: 'sequential',
            logSeverityLevel: 2,
            intraOpNumThreads: 0,
            interOpNumThreads: 0,
        }
    },
    RECOGNITION: {
        LANGUAGES: {
            en: { 
                MODEL: './models/en_PP-OCRv4_rec_infer.onnx',
                DICT: './models/en_dict.txt',
                NAME: 'English'
            },
            ch: { 
                MODEL: './models/ch_PP-OCRv4_rec_infer.onnx',
                DICT: './models/ch_dict.txt', 
                NAME: 'Chinese'
            },
            ja: { 
                MODEL: './models/japan_PP-OCRv3_rec_infer.onnx',
                DICT: './models/japan_dict.txt',
                NAME: 'Japanese'
            },
            ko: { 
                MODEL: './models/ch_PP-OCRv4_rec_infer.onnx',
                DICT: './models/korean_dict.txt',
                NAME: 'Korean'
            },
            latin: { 
                MODEL: './models/latin_PP-OCRv3_rec_infer.onnx',
                DICT: './models/latin_dict.txt',
                NAME: 'Latin'
            }
        },
        DEFAULT_LANGUAGE: 'en',
        IMAGE_HEIGHT: 48,
        CONFIDENCE_THRESHOLD: 0.5,
        REMOVE_DUPLICATE_CHARS: true,
        IGNORED_TOKENS: [0],
        ONNX_OPTIONS: {
            executionProviders: ['cpu'],
            graphOptimizationLevel: 'all',
            enableCpuMemArena: true,
            enableMemPattern: true,
            executionMode: 'sequential',
            logSeverityLevel: 2,
            intraOpNumThreads: 0,
            interOpNumThreads: 0,
        }
    },
    GROUPING: {
        VERTICAL_THRESHOLD_RATIO: 1.2,
        HORIZONTAL_THRESHOLD_RATIO: 2.5,
        MIN_OVERLAP_RATIO: 0.3,
        MAX_VERTICAL_OFFSET_RATIO: 0.5,
    }
};

// =============================================================================
// TEXT GROUPING FUNCTIONS
// =============================================================================

function calculateDistance(box1, box2) {
    const center1 = {
        x: box1.left + box1.width / 2,
        y: box1.top + box1.height / 2
    };
    const center2 = {
        x: box2.left + box2.width / 2,
        y: box2.top + box2.height / 2
    };
    
    return {
        horizontal: Math.abs(center1.x - center2.x),
        vertical: Math.abs(center1.y - center2.y),
        euclidean: Math.sqrt(Math.pow(center1.x - center2.x, 2) + Math.pow(center1.y - center2.y, 2))
    };
}

function boxesOverlapVertically(box1, box2) {
    const top1 = box1.top;
    const bottom1 = box1.top + box1.height;
    const top2 = box2.top;
    const bottom2 = box2.top + box2.height;
    
    const overlapTop = Math.max(top1, top2);
    const overlapBottom = Math.min(bottom1, bottom2);
    const overlap = Math.max(0, overlapBottom - overlapTop);
    
    const minHeight = Math.min(box1.height, box2.height);
    return overlap / minHeight;
}

function areOnSameLine(box1, box2, config) {
    const overlapRatio = boxesOverlapVertically(box1, box2);
    const verticalOffset = Math.abs((box1.top + box1.height / 2) - (box2.top + box2.height / 2));
    const avgHeight = (box1.height + box2.height) / 2;
    
    return overlapRatio >= config.MIN_OVERLAP_RATIO || 
           verticalOffset < avgHeight * config.MAX_VERTICAL_OFFSET_RATIO;
}

function shouldGroup(box1, box2, avgHeight, config) {
    const distance = calculateDistance(box1, box2);
    
    if (areOnSameLine(box1, box2, config)) {
        const maxHorizontalGap = avgHeight * config.HORIZONTAL_THRESHOLD_RATIO;
        const isRightOf = box2.left > box1.left;
        return isRightOf && distance.horizontal < maxHorizontalGap;
    } else {
        const maxVerticalGap = avgHeight * config.VERTICAL_THRESHOLD_RATIO;
        const horizontalOverlap = Math.min(box1.left + box1.width, box2.left + box2.width) - 
                                 Math.max(box1.left, box2.left);
        const isBelow = box2.top > box1.top;
        const hasHorizontalOverlap = horizontalOverlap > 0;
        return isBelow && distance.vertical < maxVerticalGap && hasHorizontalOverlap;
    }
}

function groupTextElements(elements, config = DEFAULT_CONFIG.GROUPING) {
    if (elements.length === 0) return [];
    
    const avgHeight = elements.reduce((sum, el) => sum + el.frame.height, 0) / elements.length;
    
    const sorted = [...elements].sort((a, b) => {
        const verticalDiff = a.frame.top - b.frame.top;
        if (Math.abs(verticalDiff) < avgHeight * 0.5) {
            return a.frame.left - b.frame.left;
        }
        return verticalDiff;
    });
    
    const groups = [];
    const used = new Set();
    
    for (let i = 0; i < sorted.length; i++) {
        if (used.has(i)) continue;
        
        const group = [i];
        used.add(i);
        
        let changed = true;
        while (changed) {
            changed = false;
            
            for (let j = 0; j < sorted.length; j++) {
                if (used.has(j)) continue;
                
                for (const groupIdx of group) {
                    if (shouldGroup(sorted[groupIdx].frame, sorted[j].frame, avgHeight, config)) {
                        group.push(j);
                        used.add(j);
                        changed = true;
                        break;
                    }
                }
                
                if (changed) break;
            }
        }
        
        group.sort((a, b) => {
            const elemA = sorted[a].frame;
            const elemB = sorted[b].frame;
            const verticalDiff = elemA.top - elemB.top;
            
            if (Math.abs(verticalDiff) < avgHeight * 0.5) {
                return elemA.left - elemB.left;
            }
            return verticalDiff;
        });
        
        groups.push(group.map(idx => sorted[idx]));
    }
    
    return groups;
}

function createParagraph(group) {
    const texts = group.map(el => el.text);
    const avgConfidence = group.reduce((sum, el) => sum + el.confidence, 0) / group.length;
    
    const allX = group.flatMap(el => [el.frame.left, el.frame.left + el.frame.width]);
    const allY = group.flatMap(el => [el.frame.top, el.frame.top + el.frame.height]);
    
    const boundingBox = {
        left: Math.min(...allX),
        top: Math.min(...allY),
        width: Math.max(...allX) - Math.min(...allX),
        height: Math.max(...allY) - Math.min(...allY)
    };
    
    return {
        text: texts.join(' '),
        confidence: avgConfidence,
        boundingBox,
        elements: group.map(el => ({
            text: el.text,
            confidence: el.confidence,
            frame: el.frame
        }))
    };
}

// =============================================================================
// UTILITY CLASSES
// =============================================================================

class FileUtils {
    static async read(filePath) {
        return await fs.readFile(filePath, 'utf8');
    }
}

class ImageRaw {
    data;
    width;
    height;
    #sharp;

    static async open(filePath) {
        const sharpInstance = sharp(filePath).ensureAlpha();
        const result = await sharpInstance.raw().toBuffer({ resolveWithObject: true });
        return new ImageRaw({
            data: result.data,
            width: result.info.width,
            height: result.info.height
        });
    }

    constructor(imageRawData) {
        this.data = imageRawData.data;
        this.width = imageRawData.width;
        this.height = imageRawData.height;
        this.#sharp = sharp(imageRawData.data, {
            raw: { width: imageRawData.width, height: imageRawData.height, channels: 4 }
        });
    }

    async resize(size) {
        const resized = await this.#sharp.resize({
            width: size.width,
            height: size.height,
            fit: 'contain'
        }).raw().toBuffer({ resolveWithObject: true });
        
        this.data = resized.data;
        this.width = resized.info.width;
        this.height = resized.info.height;
        this.#sharp = sharp(this.data, {
            raw: { width: this.width, height: this.height, channels: 4 }
        });
        return this;
    }
}

class ModelBase {
    options;
    #model;

    constructor({ model, options }) {
        this.#model = model;
        this.options = options;
    }

    async runModel({ modelData, onnxOptions = {} }) {
        const input = new Tensor('float32', Float32Array.from(modelData.data), 
            [1, 3, modelData.height, modelData.width]);
        const outputs = await this.#model.run({
            [this.#model.inputNames[0]]: input
        }, onnxOptions);
        return outputs[this.#model.outputNames[0]];
    }

    imageToInput(image, { mean = [0, 0, 0], std = [1, 1, 1] } = {}) {
        const R = [], G = [], B = [];
        for (let i = 0; i < image.data.length; i += 4) {
            R.push((image.data[i] / 255 - mean[0]) / std[0]);
            G.push((image.data[i + 1] / 255 - mean[1]) / std[1]);
            B.push((image.data[i + 2] / 255 - mean[2]) / std[2]);
        }
        return {
            data: [...B, ...G, ...R],
            width: image.width,
            height: image.height
        };
    }
}

// =============================================================================
// OPENCV HELPER FUNCTIONS
// =============================================================================

function cvImread(image) {
    const mat = new cv.Mat(image.height, image.width, cv.CV_8UC4);
    mat.data.set(image.data);
    return mat;
}

function cvImshow(mat) {
    return new ImageRaw({ 
        data: Buffer.from(mat.data),
        width: mat.cols, 
        height: mat.rows 
    });
}

function getMiniBoxes(contour) {
    const boundingBox = cv.minAreaRect(contour);
    const points = Array.from(boxPoints(boundingBox)).sort((a, b) => a[0] - b[0]);
    
    let index_1 = 0, index_4 = 1;
    if (points[1][1] > points[0][1]) {
        index_1 = 0; index_4 = 1;
    } else {
        index_1 = 1; index_4 = 0;
    }
    
    let index_2 = 2, index_3 = 3;
    if (points[3][1] > points[2][1]) {
        index_2 = 2; index_3 = 3;
    } else {
        index_2 = 3; index_3 = 2;
    }
    
    const box = [points[index_1], points[index_2], points[index_3], points[index_4]];
    return { points: box, sside: Math.min(boundingBox.size.height, boundingBox.size.width) };
}

function boxPoints(rotatedRect) {
    const points = [];
    const angle = rotatedRect.angle * Math.PI / 180.0;
    const b = Math.cos(angle) * 0.5;
    const a = Math.sin(angle) * 0.5;
    const center = rotatedRect.center;
    const size = rotatedRect.size;
    
    points[0] = [center.x - a * size.height - b * size.width, center.y + b * size.height - a * size.width];
    points[1] = [center.x + a * size.height - b * size.width, center.y - b * size.height - a * size.width];
    points[2] = [center.x + a * size.height + b * size.width, center.y - b * size.height + a * size.width];
    points[3] = [center.x - a * size.height + b * size.width, center.y + b * size.height + a * size.width];
    
    return points;
}

function polygonPolygonArea(polygon) {
    let area = 0;
    for (let i = 0; i < polygon.length; i++) {
        const j = (i + 1) % polygon.length;
        area += polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1];
    }
    return Math.abs(area) / 2.0;
}

function polygonPolygonLength(polygon) {
    let length = 0;
    for (let i = 0; i < polygon.length; i++) {
        const j = (i + 1) % polygon.length;
        length += Math.sqrt(Math.pow(polygon[j][0] - polygon[i][0], 2) + Math.pow(polygon[j][1] - polygon[i][1], 2));
    }
    return length;
}

function orderPointsClockwise(pts) {
    const s = pts.map(pt => pt[0] + pt[1]);
    const rect = [
        pts[s.indexOf(Math.min(...s))],
        null,
        pts[s.indexOf(Math.max(...s))],
        null
    ];
    
    const tmp = pts.filter(pt => pt !== rect[0] && pt !== rect[2]);
    const diff = [tmp[0][1] - tmp[1][1], tmp[0][0] - tmp[1][0]];
    rect[1] = diff[1] > 0 ? tmp[0] : tmp[1];
    rect[3] = diff[1] > 0 ? tmp[1] : tmp[0];
    
    return rect;
}

function linalgNorm(p0, p1) {
    return Math.sqrt(Math.pow(p0[0] - p1[0], 2) + Math.pow(p0[1] - p1[1], 2));
}

function getRotateCropImage(imageRaw, points) {
    const img_crop_width = Math.floor(Math.max(linalgNorm(points[0], points[1]), linalgNorm(points[2], points[3])));
    const img_crop_height = Math.floor(Math.max(linalgNorm(points[0], points[3]), linalgNorm(points[1], points[2])));
    
    const pts_std = [[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]];
    
    const srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, points.flat());
    const dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, pts_std.flat());
    const M = cv.getPerspectiveTransform(srcTri, dstTri);
    
    const src = cvImread(imageRaw);
    const dst = new cv.Mat();
    cv.warpPerspective(src, dst, M, new cv.Size(img_crop_width, img_crop_height), 
        cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar());
    
    let dst_rot = dst;
    if (dst.rows / dst.cols >= 1.5) {
        dst_rot = new cv.Mat();
        const M_rot = cv.getRotationMatrix2D(new cv.Point(dst.cols / 2, dst.cols / 2), 90, 1);
        cv.warpAffine(dst, dst_rot, M_rot, new cv.Size(dst.rows, dst.cols), 
            cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar());
        dst.delete();
    }
    
    src.delete();
    srcTri.delete();
    dstTri.delete();
    
    return cvImshow(dst_rot);
}

function unclip(box, unclip_ratio = 1.5) {
    const area = Math.abs(polygonPolygonArea(box));
    const length = polygonPolygonLength(box);
    const distance = (area * unclip_ratio) / length;
    const tmpArr = box.map(item => ({ X: item[0], Y: item[1] }));
    const offset = new clipper.ClipperOffset();
    offset.AddPath(tmpArr, clipper.JoinType.jtRound, clipper.EndType.etClosedPolygon);
    const expanded = [];
    offset.Execute(expanded, distance);
    return expanded[0] ? expanded[0].map(item => [item.X, item.Y]).flat() : [];
}

// =============================================================================
// DETECTION MODEL
// =============================================================================

class Detection extends ModelBase {
    static async create(options = {}) {
        const config = { ...DEFAULT_CONFIG.DETECTION, ...options };
        const model = await InferenceSession.create(config.MODEL_PATH, config.ONNX_OPTIONS);
        return new Detection({ model, config });
    }

    constructor({ model, config }) {
        super({ model, options: {} });
        this.threshold = config.THRESHOLD;
        this.minSize = config.MIN_BOX_SIZE;
        this.maxSize = config.MAX_BOX_SIZE;
        this.unclipRatio = config.UNCLIP_RATIO;
        this.baseSize = config.BASE_SIZE;
        this.maxImageSize = config.MAX_IMAGE_SIZE;
    }

    async run(imagePath, { onnxOptions = {} } = {}) {
        const image = await ImageRaw.open(imagePath);
        const inputImage = await image.resize(this.multipleOfBaseSize(image));
        const modelData = this.imageToInput(inputImage);
        const modelOutput = await this.runModel({ modelData, onnxOptions });
        const outputImage = this.outputToImage(modelOutput, this.threshold);
        return await this.splitIntoLineImages(outputImage, inputImage);
    }

    multipleOfBaseSize(image) {
        let width = image.width, height = image.height;
        if (this.maxImageSize && Math.max(width, height) > this.maxImageSize) {
            const ratio = width > height ? this.maxImageSize / width : this.maxImageSize / height;
            width *= ratio; height *= ratio;
        }
        return {
            width: Math.max(Math.ceil(width / this.baseSize) * this.baseSize, this.baseSize),
            height: Math.max(Math.ceil(height / this.baseSize) * this.baseSize, this.baseSize)
        };
    }

    outputToImage(output, threshold) {
        const [height, width] = [output.dims[2], output.dims[3]];
        const data = new Uint8Array(width * height * 4);
        output.data.forEach((outValue, outIndex) => {
            const n = outIndex * 4;
            const value = outValue > threshold ? 255 : 0;
            data[n] = data[n + 1] = data[n + 2] = value;
            data[n + 3] = 255;
        });
        return new ImageRaw({ data, width, height });
    }

    async splitIntoLineImages(image, sourceImage) {
        const src = cvImread(image);
        cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
        const contours = new cv.MatVector();
        const hierarchy = new cv.Mat();
        cv.findContours(src, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
        
        const edgeRect = [];
        const [w, h] = [image.width, image.height];
        const [rx, ry] = [sourceImage.width / w, sourceImage.height / h];
        
        for (let i = 0; i < contours.size(); i++) {
            const { points, sside } = getMiniBoxes(contours.get(i));
            if (sside < this.minSize || sside > this.maxSize) continue;
            
            const clipBox = unclip(points, this.unclipRatio);
            const boxMap = cv.matFromArray(clipBox.length / 2, 1, cv.CV_32SC2, clipBox);
            const resultObj = getMiniBoxes(boxMap);
            if (resultObj.sside < this.minSize + 2) continue;
            
            const box = resultObj.points.map(p => [p[0] * rx, p[1] * ry]);
            const box1 = orderPointsClockwise(box).map(item => [
                Math.max(0, Math.min(Math.round(item[0]), sourceImage.width)),
                Math.max(0, Math.min(Math.round(item[1]), sourceImage.height))
            ]);
            
            const rect_width = Math.floor(linalgNorm(box1[0], box1[1]));
            const rect_height = Math.floor(linalgNorm(box1[0], box1[3]));
            if (rect_width > 3 && rect_height > 3) {
                edgeRect.push({ box: box1, image: getRotateCropImage(sourceImage, box1) });
            }
        }
        
        src.delete();
        contours.delete();
        hierarchy.delete();
        return edgeRect;
    }
}

// =============================================================================
// RECOGNITION MODEL
// =============================================================================

class Recognition extends ModelBase {
    #dictionary;

    static async create(options = {}) {
        const language = options.language || DEFAULT_CONFIG.RECOGNITION.DEFAULT_LANGUAGE;
        const langConfig = DEFAULT_CONFIG.RECOGNITION.LANGUAGES[language];
        
        invariant(langConfig, `Unsupported language: ${language}. Available: ${Object.keys(DEFAULT_CONFIG.RECOGNITION.LANGUAGES).join(', ')}`);
        
        const config = {
            MODEL_PATH: options.modelPath || langConfig.MODEL,
            DICT_PATH: options.dictionaryPath || langConfig.DICT,
            CONFIDENCE_THRESHOLD: options.confidenceThreshold ?? DEFAULT_CONFIG.RECOGNITION.CONFIDENCE_THRESHOLD,
            IMAGE_HEIGHT: options.imageHeight ?? DEFAULT_CONFIG.RECOGNITION.IMAGE_HEIGHT,
            REMOVE_DUPLICATE_CHARS: options.removeDuplicateChars ?? DEFAULT_CONFIG.RECOGNITION.REMOVE_DUPLICATE_CHARS,
            ONNX_OPTIONS: { ...DEFAULT_CONFIG.RECOGNITION.ONNX_OPTIONS, ...options.onnxOptions }
        };
        
        const model = await InferenceSession.create(config.MODEL_PATH, config.ONNX_OPTIONS);
        const dictionary = [...(await FileUtils.read(config.DICT_PATH)).split('\n'), ' '];
        
        return new Recognition({ model, config, dictionary });
    }

    constructor({ model, config, dictionary }) {
        super({ model, options: {} });
        this.#dictionary = dictionary;
        this.confidenceThreshold = config.CONFIDENCE_THRESHOLD;
        this.imageHeight = config.IMAGE_HEIGHT;
        this.removeDuplicateChars = config.REMOVE_DUPLICATE_CHARS;
    }

    async run(lineImages, { onnxOptions = {} } = {}) {
        const modelDatas = await Promise.all(
            lineImages.map(li => li.image.resize({ height: this.imageHeight }).then(r => this.imageToInput(r)))
        );

        const allLines = [];
        for (const modelData of modelDatas) {
            const output = await this.runModel({ modelData, onnxOptions });
            allLines.unshift(...this.decodeText(output));
        }

        return allLines.map((line, i) => ({
            ...line,
            box: lineImages[allLines.length - i - 1].box
        })).filter(x => x.mean >= this.confidenceThreshold);
    }

    decodeText(output) {
        const line = [];
        const predLen = output.dims[2];
        
        for (let l = output.data.length - predLen * output.dims[1], ml = 0; l >= 0; l -= predLen * output.dims[1], ml++) {
            const predsIdx = [], predsProb = [];
            for (let i = l; i < l + predLen * output.dims[1]; i += predLen) {
                const tmpArr = output.data.slice(i, i + predLen);
                const tmpMax = Math.max(...tmpArr);
                predsProb.push(tmpMax);
                predsIdx.push(tmpArr.indexOf(tmpMax));
            }
            line[ml] = this.decode(predsIdx, predsProb);
        }
        return line;
    }

    decode(textIndex, textProb) {
        const charList = [], confList = [];
        for (let idx = 0; idx < textIndex.length; idx++) {
            if (DEFAULT_CONFIG.RECOGNITION.IGNORED_TOKENS.includes(textIndex[idx])) continue;
            if (this.removeDuplicateChars && idx > 0 && textIndex[idx - 1] === textIndex[idx]) continue;
            const char = this.#dictionary[textIndex[idx] - 1];
            if (char) {
                charList.push(char);
                confList.push(textProb[idx]);
            }
        }
        
        const text = charList.join('').replace(/\r/g, '').replace(/\s+/g, ' ').trim();
        const mean = confList.length ? confList.reduce((a, b) => a + b) / confList.length : 0;
        return { text, mean };
    }
}

// =============================================================================
// MAIN OCR CLASS
// =============================================================================

/**
 * PureJS OCR - JavaScript-only OCR solution with multilingual support
 * No Python dependencies required!
 */
class Ocr {
    #detection;
    #recognition;
    #groupingConfig;

    /**
     * Create an OCR instance
     * @param {Object} options - Configuration options
     * @param {string} options.language - Language code (en, ch, ja, ko, latin)
     * @param {number} options.detectionThreshold - Text detection threshold (0-1)
     * @param {number} options.confidenceThreshold - Recognition confidence threshold (0-1)
     * @param {number} options.minBoxSize - Minimum text box size
     * @param {number} options.maxBoxSize - Maximum text box size
     * @param {number} options.unclipRatio - Box expansion ratio
     * @param {string} options.detectionModelPath - Custom detection model path
     * @param {string} options.recognitionModelPath - Custom recognition model path
     * @param {string} options.dictionaryPath - Custom dictionary path
     * @param {Object} options.grouping - Text grouping configuration
     * @param {Object} options.detectionOnnxOptions - ONNX runtime options for detection
     * @param {Object} options.recognitionOnnxOptions - ONNX runtime options for recognition
     */
    static async create(options = {}) {
        const detectionConfig = {
            MODEL_PATH: options.detectionModelPath || DEFAULT_CONFIG.DETECTION.MODEL_PATH,
            THRESHOLD: options.detectionThreshold ?? DEFAULT_CONFIG.DETECTION.THRESHOLD,
            MIN_BOX_SIZE: options.minBoxSize ?? DEFAULT_CONFIG.DETECTION.MIN_BOX_SIZE,
            MAX_BOX_SIZE: options.maxBoxSize ?? DEFAULT_CONFIG.DETECTION.MAX_BOX_SIZE,
            UNCLIP_RATIO: options.unclipRatio ?? DEFAULT_CONFIG.DETECTION.UNCLIP_RATIO,
            BASE_SIZE: options.baseSize ?? DEFAULT_CONFIG.DETECTION.BASE_SIZE,
            MAX_IMAGE_SIZE: options.maxImageSize ?? DEFAULT_CONFIG.DETECTION.MAX_IMAGE_SIZE,
            ONNX_OPTIONS: { ...DEFAULT_CONFIG.DETECTION.ONNX_OPTIONS, ...options.detectionOnnxOptions }
        };

        const recognitionConfig = {
            language: options.language || DEFAULT_CONFIG.RECOGNITION.DEFAULT_LANGUAGE,
            modelPath: options.recognitionModelPath,
            dictionaryPath: options.dictionaryPath,
            confidenceThreshold: options.confidenceThreshold ?? DEFAULT_CONFIG.RECOGNITION.CONFIDENCE_THRESHOLD,
            imageHeight: options.imageHeight ?? DEFAULT_CONFIG.RECOGNITION.IMAGE_HEIGHT,
            removeDuplicateChars: options.removeDuplicateChars ?? DEFAULT_CONFIG.RECOGNITION.REMOVE_DUPLICATE_CHARS,
            onnxOptions: options.recognitionOnnxOptions
        };

        const groupingConfig = { ...DEFAULT_CONFIG.GROUPING, ...options.grouping };

        const detection = await Detection.create(detectionConfig);
        const recognition = await Recognition.create(recognitionConfig);
        
        return new Ocr({ detection, recognition, groupingConfig });
    }

    constructor({ detection, recognition, groupingConfig }) {
        this.#detection = detection;
        this.#recognition = recognition;
        this.#groupingConfig = groupingConfig;
    }

    /**
     * Detect and recognize text in an image
     * @param {string} imagePath - Path to the image file
     * @param {Object} options - Detection options
     * @param {boolean} options.grouped - Return grouped paragraphs (default: true)
     * @param {Object} options.onnxOptions - ONNX runtime options
     * @returns {Promise<Object>} OCR results with texts and paragraphs
     */
    async detect(imagePath, options = {}) {
        const grouped = options.grouped !== false;
        
        const lineImages = await this.#detection.run(imagePath, options);
        const texts = await this.#recognition.run(lineImages, options);
        
        const individualElements = texts
            .filter(item => item?.text && item.text.trim().length > 0)
            .map(item => ({
                text: item.text.trim(),
                confidence: item.mean,
                frame: this.extractFrameFromBox(item.box),
                box: item.box
            }));

        const result = {
            totalElements: individualElements.length,
            data: individualElements
        };

        if (grouped) {
            const groups = groupTextElements(individualElements, this.#groupingConfig);
            const paragraphs = groups.map(createParagraph);
            result.totalParagraphs = paragraphs.length;
            result.paragraphs = paragraphs;
        }

        return result;
    }

    /**
     * Update grouping configuration
     * @param {Object} config - New grouping configuration
     */
    setGroupingConfig(config) {
        this.#groupingConfig = { ...this.#groupingConfig, ...config };
    }

    /**
     * Get current grouping configuration
     * @returns {Object} Current grouping configuration
     */
    getGroupingConfig() {
        return { ...this.#groupingConfig };
    }

    /**
     * Get available languages
     * @returns {Object} Available languages with their configurations
     */
    static getAvailableLanguages() {
        return { ...DEFAULT_CONFIG.RECOGNITION.LANGUAGES };
    }

    extractFrameFromBox(box) {
        if (!box?.length) return { left: 0, top: 0, width: 0, height: 0 };
        const xs = box.map(p => p[0]);
        const ys = box.map(p => p[1]);
        return {
            left: Math.round(Math.min(...xs)),
            top: Math.round(Math.min(...ys)),
            width: Math.round(Math.max(...xs) - Math.min(...xs)),
            height: Math.round(Math.max(...ys) - Math.min(...ys))
        };
    }
}

// =============================================================================
// EXPORTS
// =============================================================================

export default Ocr;
export { Ocr, DEFAULT_CONFIG };