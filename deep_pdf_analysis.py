#!/usr/bin/env python3
"""
Deep analysis of problematic PDF to understand its structure
"""

import requests
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, TableFormerMode
from docling.document_converter import PdfFormatOption

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_pdf_with_different_ocr_engines():
    """Try different OCR engines and configurations"""
    
    test_url = "https://docs.cpuc.ca.gov/PublishedDocs/EFILE/CM/162841.PDF"
    
    print("🔬 DEEP PDF ANALYSIS WITH MULTIPLE OCR ENGINES")
    print("=" * 70)
    
    # Try different OCR configurations
    ocr_configs = [
        ("EasyOCR", lambda: EasyOcrOptions()),
        ("Default OCR", None),
    ]
    
    for config_name, ocr_option_func in ocr_configs:
        print(f"\n📋 Testing with {config_name}:")
        try:
            # Configure pipeline
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            
            if ocr_option_func:
                pipeline_options.ocr_options = ocr_option_func()
            
            # Create converter
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        backend=DoclingParseV4DocumentBackend,
                        pipeline_options=pipeline_options
                    )
                }
            )
            
            print(f"   ⏳ Processing with {config_name}...")
            conv_results = converter.convert_all([test_url], raises_on_error=False)
            conv_res = next(iter(conv_results), None)

            if conv_res and conv_res.status.name == 'SUCCESS':
                docling_doc = conv_res.document
                
                # Analyze all items
                total_items = 0
                text_items = 0
                content_items = 0
                
                print(f"   📊 Document structure analysis:")
                for item, level in docling_doc.iterate_items(with_groups=False):
                    total_items += 1
                    
                    if hasattr(item, 'text') and item.text:
                        text_items += 1
                        if item.text.strip():
                            content_items += 1
                            if content_items <= 3:  # Show first 3 text items
                                print(f"      Text item {content_items}: {repr(item.text[:100])}")
                
                print(f"   📈 Results: {total_items} items, {text_items} with text, {content_items} with content")
                
                if content_items > 0:
                    print(f"   ✅ {config_name} found extractable content!")
                    return True
                else:
                    print(f"   ❌ {config_name} found no extractable content")
            else:
                print(f"   ❌ {config_name} conversion failed")
                
        except Exception as e:
            print(f"   ❌ {config_name} failed with error: {e}")
    
    return False

def try_alternative_pdf_processing():
    """Try alternative PDF processing approaches"""
    
    test_url = "https://docs.cpuc.ca.gov/PublishedDocs/EFILE/CM/162841.PDF"
    
    print(f"\n🔄 ALTERNATIVE PDF PROCESSING APPROACHES")
    print("=" * 70)
    
    # Download PDF first to analyze locally
    try:
        print("📥 Downloading PDF for local analysis...")
        response = requests.get(test_url, timeout=60)
        pdf_path = Path("/tmp/test_162841.pdf")
        
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        
        print(f"   ✅ Downloaded {len(response.content)} bytes")
        
        # Try with local file
        print("\n📋 Testing with local file processing...")
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = EasyOcrOptions()
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # More thorough processing
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    backend=DoclingParseV4DocumentBackend,
                    pipeline_options=pipeline_options
                )
            }
        )
        
        conv_results = converter.convert_all([pdf_path], raises_on_error=False)
        conv_res = next(iter(conv_results), None)
        
        if conv_res and conv_res.status.name == 'SUCCESS':
            docling_doc = conv_res.document
            
            print(f"   📊 Local processing results:")
            
            # Try different iteration approaches
            all_text = []
            
            for item, level in docling_doc.iterate_items(with_groups=True):
                if hasattr(item, 'text') and item.text and item.text.strip():
                    all_text.append(item.text.strip())
                    if len(all_text) <= 5:  # Show first 5 texts
                        print(f"      Found text: {repr(item.text[:150])}")
            
            print(f"   📈 Total text segments found: {len(all_text)}")
            
            if all_text:
                print(f"   ✅ Local processing found extractable text!")
                return True
            else:
                print(f"   ❌ Local processing still found no text")
        else:
            print(f"   ❌ Local processing failed")
            
        # Clean up
        pdf_path.unlink()
        
    except Exception as e:
        print(f"   ❌ Alternative processing failed: {e}")
    
    return False

def check_pdf_metadata_and_structure():
    """Check PDF metadata and basic structure"""
    
    test_url = "https://docs.cpuc.ca.gov/PublishedDocs/EFILE/CM/162841.PDF"
    
    print(f"\n🔍 PDF METADATA AND STRUCTURE ANALYSIS")
    print("=" * 70)
    
    try:
        # Get PDF content
        response = requests.get(test_url, timeout=60)
        pdf_content = response.content
        
        print(f"📊 Basic PDF Information:")
        print(f"   File size: {len(pdf_content):,} bytes")
        print(f"   PDF header: {pdf_content[:50]}")
        
        # Look for common PDF issues
        if b'Linearized' in pdf_content[:1000]:
            print(f"   ✅ PDF is linearized (web-optimized)")
        
        if b'Encrypt' in pdf_content[:5000]:
            print(f"   ⚠️  PDF may be encrypted")
        
        # Count pages (rough estimate)
        page_count = pdf_content.count(b'/Type /Page')
        print(f"   📄 Estimated pages: {page_count}")
        
        # Check for common text patterns
        text_indicators = [b'BT', b'ET', b'Tj', b'TJ', b'Td', b'TD']
        text_commands = sum(1 for indicator in text_indicators if indicator in pdf_content)
        print(f"   📝 Text command indicators: {text_commands}")
        
        # Check for image indicators
        image_indicators = [b'/XObject', b'/Image', b'/DCTDecode', b'/FlateDecode']
        image_commands = sum(1 for indicator in image_indicators if indicator in pdf_content)
        print(f"   🖼️  Image indicators: {image_commands}")
        
        if image_commands > text_commands:
            print(f"   🔍 Analysis: This appears to be an image-heavy/scanned PDF")
        else:
            print(f"   🔍 Analysis: This appears to have native text content")
            
    except Exception as e:
        print(f"   ❌ Metadata analysis failed: {e}")

if __name__ == '__main__':
    print("🔬 DEEP PDF ANALYSIS")
    print("=" * 80)
    print("Analyzing problematic PDF with multiple approaches")
    print("=" * 80)
    
    # Run metadata analysis
    check_pdf_metadata_and_structure()
    
    # Try different OCR engines
    ocr_success = analyze_pdf_with_different_ocr_engines()
    
    # Try alternative processing
    alt_success = try_alternative_pdf_processing()
    
    overall_success = ocr_success or alt_success
    
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   OCR approach successful: {ocr_success}")
    print(f"   Alternative approach successful: {alt_success}")
    print(f"   Overall analysis: {'SUCCESS' if overall_success else 'FAILED'}")
    
    if not overall_success:
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   • This PDF may be corrupted or have non-standard structure")
        print(f"   • Consider manual inspection of the PDF")
        print(f"   • This type of PDF may need to be excluded from processing")
        print(f"   • Try a different PDF processing library as fallback")
    
    exit(0 if overall_success else 1)