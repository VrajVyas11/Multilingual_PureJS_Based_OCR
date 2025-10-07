// examples/advanced.js
import Ocr from './index.js';

async function basicExample(imagePath) {
    console.log('🎯 PureJS OCR - Basic Example\n');
    
    try {
        // Create OCR instance with English language (default)
        console.log('📝 Creating OCR instance...');
        const ocr = await Ocr.create({
            language: 'en',
            detectionThreshold: 0.1,
            confidenceThreshold: 0.5
        });
        
        console.log('✅ OCR instance created successfully!\n');
        
        console.log(`📸 Processing image: ${imagePath}\n`);
        
        const result = await ocr.detect(imagePath);
        
        // Display results
        console.log('=' .repeat(60));
        console.log('📊 RESULTS');
        console.log('=' .repeat(60));
        console.log(`Total text elements found: ${result.totalElements}`);
        console.log(`Grouped into paragraphs: ${result.totalParagraphs}\n`);
        
        // Show paragraphs
        console.log('📝 PARAGRAPHS:\n');
        result.paragraphs.forEach((para, index) => {
            console.log(`${index + 1}. "${para.text}"`);
            console.log(`   Confidence: ${(para.confidence * 100).toFixed(1)}%`);
            console.log(`   Position: (${para.boundingBox.left}, ${para.boundingBox.top})`);
            console.log(`   Size: ${para.boundingBox.width}x${para.boundingBox.height}px`);
            console.log(`   Elements: ${para.elements.length}\n`);
        });
        
        console.log('=' .repeat(60));
        console.log('🎉 Processing complete!');
        console.log('=' .repeat(60));
        
        // Export JSON
        console.log('\n💾 Full JSON output:\n');
        console.log(JSON.stringify(result, null, 2));
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
}

async function advancedExample(imagePath) {
    console.log('🚀 PureJS OCR - Advanced Configuration Example\n');
    
    try {
        // Create OCR with advanced configuration
        console.log('⚙️  Creating OCR with custom configuration...');
        const ocr = await Ocr.create({
            // Language
            language: 'en',
            
            // Detection fine-tuning
            detectionThreshold: 0.08,      // More sensitive
            minBoxSize: 5,
            maxBoxSize: 1000,
            unclipRatio: 1.8,              // More expansion
            
            // Recognition fine-tuning
            confidenceThreshold: 0.6,      // Higher confidence requirement
            imageHeight: 48,
            removeDuplicateChars: true,
            
            // Custom grouping logic
            grouping: {
                verticalThresholdRatio: 1.5,
                horizontalThresholdRatio: 3.0,
                minOverlapRatio: 0.25,
                maxVerticalOffsetRatio: 0.6
            },
            
            // Performance optimization
            detectionOnnxOptions: {
                executionProviders: ['cpu'],
                intraOpNumThreads: 4,
                graphOptimizationLevel: 'all',
                enableCpuMemArena: true
            },
            
            recognitionOnnxOptions: {
                executionProviders: ['cpu'],
                intraOpNumThreads: 2,
                executionMode: 'sequential'
            }
        });
        
        console.log('✅ OCR configured!\n');
        
        console.log(`📸 Processing: ${imagePath}\n`);
        
        // Process with grouping
        console.log('🔍 Detection with paragraph grouping...');
        const resultGrouped = await ocr.detect(imagePath, { grouped: true });
        
        console.log(`✅ Found ${resultGrouped.totalElements} elements in ${resultGrouped.totalParagraphs} paragraphs\n`);
        
        // Process without grouping
        console.log('🔍 Detection without grouping...');
        const resultUngrouped = await ocr.detect(imagePath, { grouped: false });
        
        console.log(`✅ Found ${resultUngrouped.totalElements} individual elements\n`);
        
        // Dynamic grouping configuration
        console.log('🔧 Updating grouping configuration...');
        ocr.setGroupingConfig({
            horizontalThresholdRatio: 4.0,  // Even more horizontal tolerance
            verticalThresholdRatio: 2.0
        });
        
        const currentConfig = ocr.getGroupingConfig();
        console.log('Current grouping config:', currentConfig);
        console.log();
        
        // Show detailed results
        console.log('=' .repeat(70));
        console.log('📊 DETAILED RESULTS');
        console.log('=' .repeat(70));
        
        resultGrouped.paragraphs.forEach((para, idx) => {
            console.log(`\nParagraph ${idx + 1}:`);
            console.log(`  Text: "${para.text}"`);
            console.log(`  Confidence: ${(para.confidence * 100).toFixed(2)}%`);
            console.log(`  Bounding Box: [${para.boundingBox.left}, ${para.boundingBox.top}, ${para.boundingBox.width}, ${para.boundingBox.height}]`);
            console.log(`  Sub-elements:`);
            para.elements.forEach((el, i) => {
                console.log(`    ${i + 1}. "${el.text}" (${(el.confidence * 100).toFixed(1)}%)`);
            });
        });
        
        console.log('\n' + '=' .repeat(70));
        console.log('✨ Advanced processing complete!');
        console.log('=' .repeat(70));
        
        // Performance metrics
        console.log('\n📈 Performance Metrics:');
        console.log(`  Total elements detected: ${resultGrouped.totalElements}`);
        console.log(`  Average confidence: ${(resultGrouped.data.reduce((sum, el) => sum + el.confidence, 0) / resultGrouped.data.length * 100).toFixed(2)}%`);
        console.log(`  Paragraph compression: ${((1 - resultGrouped.totalParagraphs / resultGrouped.totalElements) * 100).toFixed(1)}%`);
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
}

// Parse command-line arguments
const args = process.argv.slice(2);
const imagePath = args.find(arg => !arg.startsWith('--')) || './bs.jpeg';
const runBasic = args.includes('--basic');
const runAdvanced = args.includes('--advanced');

// Run examples based on flags
(async () => {
    if (runBasic) {
        await basicExample(imagePath);
    }
    if (runAdvanced) {
        await advancedExample(imagePath);
    }
    if (!runBasic && !runAdvanced) {
        // Default: run both
        await basicExample(imagePath);
        console.log('\n' + '=' .repeat(60));  // Separator between examples
        await advancedExample(imagePath);
    }
})();