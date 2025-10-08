# Multilingual PureJS OCR 🚀

[![npm version](https://badge.fury.io/js/multilingual-purejs-ocr.svg)](https://www.npmjs.com/package/multilingual-purejs-ocr)

A powerful onnxruntime-node based, **pure JavaScript OCR library** that works directly in Node.js - **no Python dependencies required!** Extract text from images with industry-leading accuracy using PaddleOCR models.

## ✨ Features

- 🌍 **Multilingual Support**: English, Chinese, Japanese, Korean, and Latin languages
- 🎯 **High Accuracy**: Based on PaddleOCR v3/v4 models
- 📦 **Zero Python Dependencies**: 100% JavaScript/Node.js implementation
- 🧠 **ONNX Runtime Node**: Efficient, high-performance model inference directly in Node.js
- 🔧 **Highly Customizable**: Configure detection, recognition, and grouping parameters
- 🎨 **Smart Text Grouping**: Automatically groups text into paragraphs
- 🔌 **Bring Your Own Models**: Use custom ONNX models
- ⚡ **Production Ready**: Built with performance and reliability in mind
- 🚀 **Plug and Play**: All models and dictionaries included - just install and run!

## 📥 Installation

```bash
npm install multilingual-purejs-ocr
```

## 🚀 Quick Start

```javascript
import Ocr from 'multilingual-purejs-ocr';

// Create OCR instance with default settings (English)
const ocr = await Ocr.create();

// Detect text in an image
const result = await ocr.detect('./image.jpg');

console.log('Total elements:', result.totalElements);
console.log('Paragraphs:', result.paragraphs);
console.log('Individual elements:', result.data);
```

## 📖 Usage Examples

### Basic Usage - Different Languages

```javascript
// English (default)
const ocrEn = await Ocr.create({ language: 'en' });
const resultEn = await ocrEn.detect('./english.jpg');

// Chinese
const ocrCh = await Ocr.create({ language: 'ch' });
const resultCh = await ocrCh.detect('./chinese.jpg');

// Japanese
const ocrJa = await Ocr.create({ language: 'ja' });
const resultJa = await ocrJa.detect('./japanese.jpg');

// Korean
const ocrKo = await Ocr.create({ language: 'ko' });
const resultKo = await ocrKo.detect('./korean.jpg');

// Latin languages (Portuguese, Spanish, French, etc.)
const ocrLatin = await Ocr.create({ language: 'latin' });
const resultLatin = await ocrLatin.detect('./latin.jpg');
```

### Advanced Configuration

```javascript
const ocr = await Ocr.create({
  // Language selection
  language: 'en',
  
  // Detection parameters
  detectionThreshold: 0.1,      // Lower = more sensitive (0-1)
  minBoxSize: 3,                // Minimum text box size in pixels
  maxBoxSize: 2000,             // Maximum text box size in pixels
  unclipRatio: 1.5,             // Box expansion ratio
  
  // Recognition parameters
  confidenceThreshold: 0.5,     // Minimum confidence to accept text (0-1)
  imageHeight: 48,              // Recognition model input height
  removeDuplicateChars: true,   // Remove duplicate characters
  
  // Text grouping configuration
  grouping: {
    verticalThresholdRatio: 1.2,
    horizontalThresholdRatio: 2.5,
    minOverlapRatio: 0.3,
    maxVerticalOffsetRatio: 0.5
  }
});

const result = await ocr.detect('./image.jpg');
```

### Using Custom Models

```javascript
const ocr = await Ocr.create({
  language: 'en',
  detectionModelPath: './my-models/custom_detection.onnx',
  recognitionModelPath: './my-models/custom_recognition.onnx',
  dictionaryPath: './my-models/custom_dict.txt'
});
```

### Performance Tuning with ONNX Options

```javascript
const ocr = await Ocr.create({
  language: 'en',
  
  // Detection model ONNX options
  detectionOnnxOptions: {
    executionProviders: ['cpu'],
    intraOpNumThreads: 4,          // Use 4 CPU threads
    graphOptimizationLevel: 'all',
    enableCpuMemArena: true
  },
  
  // Recognition model ONNX options
  recognitionOnnxOptions: {
    executionProviders: ['cpu'],
    intraOpNumThreads: 2,
    executionMode: 'sequential'
  }
});
```

### Without Text Grouping

```javascript
const ocr = await Ocr.create({ language: 'en' });

// Get only individual text elements (no paragraph grouping)
const result = await ocr.detect('./image.jpg', { grouped: false });

console.log(result.data); // Array of individual text elements
// result.paragraphs will not be present
```

### Dynamic Grouping Configuration

```javascript
const ocr = await Ocr.create({ language: 'en' });

// Update grouping configuration at runtime
ocr.setGroupingConfig({
  verticalThresholdRatio: 1.5,
  horizontalThresholdRatio: 3.0
});

// Get current configuration
const config = ocr.getGroupingConfig();
console.log(config);
```

## 📊 Output Format

### Complete Response Structure

```javascript
{
  totalElements: 31,           // Total text elements detected
  totalParagraphs: 9,          // Number of grouped paragraphs
  
  // Grouped paragraphs (when grouped: true)
  paragraphs: [
    {
      text: "BECAUSE OF YOU,",
      confidence: 0.923,
      boundingBox: {
        left: 485,
        top: 61,
        width: 84,
        height: 43
      },
      elements: [              // Individual elements in this paragraph
        {
          text: "BECAUSE",
          confidence: 0.96,
          frame: { left: 485, top: 61, width: 84, height: 20 }
        },
        {
          text: "OF YOU,",
          confidence: 0.86,
          frame: { left: 489, top: 81, width: 75, height: 23 }
        }
      ]
    }
  ],
  
  // Individual text elements
  data: [
    {
      text: "BECAUSE",
      confidence: 0.9598,
      frame: {
        left: 485,
        top: 61,
        width: 84,
        height: 20
      },
      box: [[485, 61], [569, 61], [569, 81], [485, 81]]  // 4-point polygon
    }
  ]
}
```

## 🎛️ Configuration Options

### Detection Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `detectionThreshold` | number | 0.1 | Text detection sensitivity (0-1, lower = more sensitive) |
| `minBoxSize` | number | 3 | Minimum text box size in pixels |
| `maxBoxSize` | number | 2000 | Maximum text box size in pixels |
| `unclipRatio` | number | 1.5 | Box expansion ratio for better text capture |
| `baseSize` | number | 32 | Image size base unit (must be multiple of 32) |
| `maxImageSize` | number | 960 | Maximum image dimension |

### Recognition Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `language` | string | 'en' | Language code: en, ch, ja, ko, latin |
| `confidenceThreshold` | number | 0.5 | Minimum confidence to accept text (0-1) |
| `imageHeight` | number | 48 | Recognition model input height |
| `removeDuplicateChars` | boolean | true | Remove consecutive duplicate characters |

### Grouping Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `verticalThresholdRatio` | number | 1.2 | Max vertical gap ratio for grouping |
| `horizontalThresholdRatio` | number | 2.5 | Max horizontal gap ratio for same line |
| `minOverlapRatio` | number | 0.3 | Minimum vertical overlap for same line |
| `maxVerticalOffsetRatio` | number | 0.5 | Max vertical offset ratio for same line |

## 🌍 Supported Languages

```javascript
// Get list of available languages
const languages = Ocr.getAvailableLanguages();
console.log(languages);
```

| Code | Language | Model |
|------|----------|-------|
| `en` | English | PP-OCRv4 |
| `ch` | Chinese | PP-OCRv4 |
| `ja` | Japanese | PP-OCRv3 |
| `ko` | Korean | PP-OCRv4 (Korean dict) |
| `latin` | Latin (PT, ES, FR, etc.) | PP-OCRv3 |

## 📁 Included Models

The package ships with all required models and dictionaries for plug-and-play usage:

```
models/
├── ch_PP-OCRv4_det_infer.onnx         # Detection model (shared)
├── en_PP-OCRv4_rec_infer.onnx         # English recognition
├── en_dict.txt                         # English dictionary
├── ch_PP-OCRv4_rec_infer.onnx         # Chinese recognition
├── ch_dict.txt                         # Chinese dictionary
├── japan_PP-OCRv3_rec_infer.onnx      # Japanese recognition
├── japan_dict.txt                      # Japanese dictionary
├── korean_dict.txt                     # Korean dictionary
├── latin_PP-OCRv3_rec_infer.onnx      # Latin recognition
└── latin_dict.txt                      # Latin dictionary
```

No additional setup required - everything is ready to use out of the box!

## 🔧 Custom Models

You can use your own ONNX models:

```javascript
const ocr = await Ocr.create({
  detectionModelPath: './path/to/detection.onnx',
  recognitionModelPath: './path/to/recognition.onnx',
  dictionaryPath: './path/to/dictionary.txt'
});
```

### Model Requirements

- **Detection Model**: ONNX format, input shape [1, 3, H, W], output: binary mask
- **Recognition Model**: ONNX format, input shape [1, 3, 48, W], output: character probabilities
- **Dictionary**: Text file with one character per line

## 🎯 Use Cases

- 📸 **Document Scanning**: Extract text from scanned documents
- 🎫 **Receipt Processing**: Parse receipts and invoices
- 🚗 **License Plate Recognition**: Read vehicle plates
- 📝 **Form Processing**: Extract data from forms
- 🌐 **Translation Apps**: Detect and translate text in images
- 📱 **Screenshot OCR**: Extract text from screenshots
- 🏷️ **Label Reading**: Read product labels and barcodes

## ⚡ Performance Tips

1. **Adjust Detection Threshold**: Lower for harder-to-read text, higher for cleaner images
2. **Use Appropriate Language Model**: Select the correct language for better accuracy
3. **Optimize Image Size**: Resize very large images before processing
4. **Tune ONNX Threads**: Adjust `intraOpNumThreads` based on your CPU cores
5. **Disable Grouping**: Use `grouped: false` if you only need individual elements

## 🐛 Troubleshooting

### Low Accuracy

- Adjust `detectionThreshold` (try 0.05 for harder images)
- Increase `unclipRatio` (try 1.8 or 2.0)
- Ensure correct language model is selected
- Preprocess image (increase contrast, remove noise)

### Missing Text

- Lower `detectionThreshold`
- Reduce `minBoxSize`
- Increase `unclipRatio`

### Performance Issues

- Reduce `maxImageSize`
- Adjust ONNX thread settings
- Process images in batches

### Wrong Grouping

- Adjust grouping ratios in configuration
- Use `grouped: false` and implement custom grouping

## 📝 API Reference

### `Ocr.create(options)`

Creates a new OCR instance with the specified configuration.

**Parameters:**
- `options` (Object): Configuration options

**Returns:** `Promise<Ocr>` - OCR instance

### `ocr.detect(imagePath, options)`

Detects and recognizes text in an image.

**Parameters:**
- `imagePath` (string): Path to image file
- `options` (Object): Detection options
  - `grouped` (boolean): Return grouped paragraphs (default: true)
  - `onnxOptions` (Object): Runtime ONNX options

**Returns:** `Promise<Object>` - Detection results

### `ocr.setGroupingConfig(config)`

Updates the text grouping configuration.

**Parameters:**
- `config` (Object): New grouping configuration

### `ocr.getGroupingConfig()`

Returns the current grouping configuration.

**Returns:** `Object` - Current grouping configuration

### `Ocr.getAvailableLanguages()`

Static method that returns available languages.

**Returns:** `Object` - Language configurations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

ISC

## 🙏 Acknowledgments

- Built on [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) models
- Uses [ONNX Runtime Node](https://www.npmjs.com/package/onnxruntime-node) for efficient model inference in Node.js
- Powered by [Sharp](https://sharp.pixelplumbing.com/) for image processing

## 📧 Support

For issues and questions, please open an issue on [GitHub](https://github.com/VrajVyas11/Multilingual_PureJS_Based_OCR/issues).

---

Made with ❤️ for the JavaScript community