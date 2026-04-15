"""
OCR Database Manager
Handles persistent storage of OCR results with frame images and wagon tracking
"""

import json
import os
from datetime import datetime
from pathlib import Path
import cv2
import re
from difflib import SequenceMatcher


class OCRDatabase:
    def __init__(self, db_path="ocr_logs"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.frames_path = self.db_path / "frames"
        self.frames_path.mkdir(exist_ok=True)
        self.reports_path = self.db_path / "reports"
        self.reports_path.mkdir(exist_ok=True)
        self.index_file = self.db_path / "index.json"
        self.load_index()
    
    def load_index(self):
        """Load the index of all processed videos"""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
        else:
            self.index = {
                "videos": [],
                "total_scans": 0,
                "last_updated": None
            }
    
    def save_index(self):
        """Save the index"""
        self.index["last_updated"] = datetime.now().isoformat()
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)
    
    def create_video_log(self, video_name):
        """Create a new log entry for a video"""
        video_id = f"{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        video_entry = {
            "video_id": video_id,
            "video_name": video_name,
            "timestamp": datetime.now().isoformat(),
            "frames_with_text": [],
            "text_with_numbers": [],
            "text_only": [],
            "wagons": [],  # New: Track individual wagons
            "total_frames_processed": 0,
            "total_text_detected": 0,
            "total_wagons": 0
        }
        
        self.index["videos"].append(video_entry)
        self.index["total_scans"] += 1
        self.save_index()
        
        return video_id
    
    def has_numbers(self, text):
        """Check if text contains numbers"""
        return bool(re.search(r'\d', text))
    
    def count_digits(self, text):
        """Count the number of digits in text"""
        return len(re.findall(r'\d', text))
    
    def has_invalid_symbols(self, text):
        """Check if text contains invalid symbols (., ,, %)"""
        invalid_symbols = ['.', ',', '%']
        return any(symbol in text for symbol in invalid_symbols)
    
    def is_wagon_number(self, text):
        """
        Check if text qualifies as a wagon number:
        - Must have 5 or more digits
        - Cannot contain . , % symbols
        - Can contain / - symbols
        """
        digit_count = self.count_digits(text)
        has_invalid = self.has_invalid_symbols(text)
        
        return digit_count >= 5 and not has_invalid
    
    def filter_text(self, text):
        """Filter text - must be 3+ characters"""
        return len(text.strip()) >= 3
    
    def text_similarity(self, text1, text2):
        """Calculate similarity between two texts (0-1)"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def find_matching_wagon(self, video_entry, wagon_number, similarity_threshold=0.85):
        """
        Find if a wagon with similar number already exists
        Returns wagon index if found, None otherwise
        """
        for idx, wagon in enumerate(video_entry['wagons']):
            similarity = self.text_similarity(wagon['wagon_number'], wagon_number)
            # If similarity is high (allowing 2-3 character difference)
            if similarity >= similarity_threshold:
                return idx
        return None
    
    def add_ocr_result(self, video_id, frame_number, frame_image, text_results):
        """Add OCR results for a frame with wagon tracking"""
        # Find video entry
        video_entry = None
        for video in self.index["videos"]:
            if video["video_id"] == video_id:
                video_entry = video
                break
        
        if not video_entry:
            return
        
        # Filter and categorize text
        valid_texts = []
        wagon_numbers = []  # Text with 5+ digits, no invalid symbols (potential wagon IDs)
        other_texts = []    # Text without enough numbers
        
        for result in text_results:
            text = result['text'].strip()
            confidence = result['confidence']
            
            # Filter short text
            if not self.filter_text(text):
                continue
            
            valid_texts.append({
                'text': text,
                'confidence': confidence
            })
            
            # Categorize based on digit count and symbols
            if self.is_wagon_number(text):  # Must have 5+ digits, no . , % symbols
                wagon_numbers.append({
                    'text': text,
                    'confidence': confidence
                })
                
                if text not in [item['text'] for item in video_entry['text_with_numbers']]:
                    video_entry['text_with_numbers'].append({
                        'text': text,
                        'confidence': confidence,
                        'first_seen_frame': frame_number
                    })
            elif self.has_numbers(text):
                # Has numbers but doesn't qualify as wagon number
                if text not in [item['text'] for item in video_entry['text_with_numbers']]:
                    video_entry['text_with_numbers'].append({
                        'text': text,
                        'confidence': confidence,
                        'first_seen_frame': frame_number
                    })
            else:
                other_texts.append({
                    'text': text,
                    'confidence': confidence
                })
                
                if text not in [item['text'] for item in video_entry['text_only']]:
                    video_entry['text_only'].append({
                        'text': text,
                        'confidence': confidence,
                        'first_seen_frame': frame_number
                    })
        
        # Wagon tracking: Only track if we have valid wagon numbers (5+ digits, no invalid symbols)
        if wagon_numbers:
            # Use the first (usually most prominent) wagon number
            primary_wagon_number = wagon_numbers[0]['text']
            
            # Check if this wagon already exists
            wagon_idx = self.find_matching_wagon(video_entry, primary_wagon_number)
            
            if wagon_idx is not None:
                # Update existing wagon
                wagon = video_entry['wagons'][wagon_idx]
                wagon['last_seen_frame'] = frame_number
                wagon['frame_count'] += 1
                
                # Add all detected texts as details
                for text_item in valid_texts:
                    if text_item['text'] not in [d['text'] for d in wagon['details']]:
                        wagon['details'].append(text_item)
                
                # Add frame reference
                wagon['frames'].append({
                    'frame_number': frame_number,
                    'texts': valid_texts
                })
            else:
                # Create new wagon
                new_wagon = {
                    'wagon_id': len(video_entry['wagons']) + 1,
                    'wagon_number': primary_wagon_number,
                    'first_seen_frame': frame_number,
                    'last_seen_frame': frame_number,
                    'frame_count': 1,
                    'details': valid_texts.copy(),  # All detected text for this wagon
                    'frames': [{
                        'frame_number': frame_number,
                        'texts': valid_texts
                    }],
                    'confidence': wagon_numbers[0]['confidence']
                }
                video_entry['wagons'].append(new_wagon)
                video_entry['total_wagons'] = len(video_entry['wagons'])
        
        # Save frame image if text was detected
        if valid_texts:
            frame_filename = f"{video_id}_frame_{frame_number}.jpg"
            frame_path = self.frames_path / frame_filename
            cv2.imwrite(str(frame_path), frame_image)
            
            video_entry['frames_with_text'].append({
                'frame_number': frame_number,
                'frame_image': frame_filename,
                'texts': valid_texts
            })
            
            video_entry['total_text_detected'] += len(valid_texts)
        
        video_entry['total_frames_processed'] = frame_number
        self.save_index()
    
    def get_all_videos(self):
        """Get all processed videos"""
        return self.index["videos"]
    
    def get_video_by_id(self, video_id):
        """Get specific video log"""
        for video in self.index["videos"]:
            if video["video_id"] == video_id:
                return video
        return None
    
    def get_frame_image_path(self, frame_filename):
        """Get full path to frame image"""
        return self.frames_path / frame_filename
    
    def truncate_wagon_number(self, wagon_number):
        """Truncate wagon number if longer than 7 digits, keep only ending"""
        if len(wagon_number) > 7:
            return "..." + wagon_number[-7:]
        return wagon_number
    
    def export_video_report(self, video_id):
        """Export wagon-focused report - one section per wagon"""
        video = self.get_video_by_id(video_id)
        if not video:
            return None
        
        report_path = self.reports_path / f"{video_id}_wagon_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"🚂 WAGON DETECTION REPORT\n")
            f.write(f"Video: {video['video_name']}\n")
            f.write(f"Scan Date: {video['timestamp'][:19]}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"📊 SUMMARY: {video.get('total_wagons', 0)} Wagon(s) Detected\n")
            f.write(f"{'─'*80}\n\n")
            
            # Main content: One section per wagon
            if video.get('wagons'):
                for idx, wagon in enumerate(video['wagons'], 1):
                    truncated_number = self.truncate_wagon_number(wagon['wagon_number'])
                    
                    f.write("="*80 + "\n")
                    f.write(f"WAGON #{idx}\n")
                    f.write("="*80 + "\n\n")
                    
                    f.write(f"Wagon Number: {truncated_number}\n")
                    if len(wagon['wagon_number']) > 7:
                        f.write(f"  (Full: {wagon['wagon_number']})\n")
                    f.write(f"Confidence: {wagon['confidence']:.2%}\n")
                    f.write(f"Frame Range: {wagon['first_seen_frame']} - {wagon['last_seen_frame']}\n")
                    f.write(f"Total Frames: {wagon['frame_count']}\n\n")
                    
                    f.write("Detected Information:\n")
                    f.write("─" * 80 + "\n")
                    
                    # Group details by type
                    numbers = [d for d in wagon['details'] if self.has_numbers(d['text'])]
                    texts = [d for d in wagon['details'] if not self.has_numbers(d['text'])]
                    
                    if numbers:
                        f.write("\n📋 Numbers/IDs:\n")
                        for detail in numbers:
                            f.write(f"  • {detail['text']:<50} (confidence: {detail['confidence']:.2%})\n")
                    
                    if texts:
                        f.write("\n📝 Text:\n")
                        for detail in texts:
                            f.write(f"  • {detail['text']:<50} (confidence: {detail['confidence']:.2%})\n")
                    
                    f.write("\n" + "─" * 80 + "\n")
                    f.write(f"Frames where this wagon was detected:\n")
                    frame_numbers = [str(frame['frame_number']) for frame in wagon['frames']]
                    f.write(f"  {', '.join(frame_numbers)}\n")
                    f.write("\n\n")
            else:
                f.write("="*80 + "\n")
                f.write("NO WAGONS DETECTED\n")
                f.write("="*80 + "\n\n")
                f.write("No valid wagon numbers found in this video.\n\n")
                f.write("Wagon Detection Criteria:\n")
                f.write("  • Must contain 5 or more digits\n")
                f.write("  • Can contain / and - symbols\n")
                f.write("  • Cannot contain . , % symbols\n\n")
            
            # Footer with criteria
            f.write("\n" + "="*80 + "\n")
            f.write("ℹ️  DETECTION CRITERIA\n")
            f.write("="*80 + "\n")
            f.write("Valid Wagon Numbers:\n")
            f.write("  ✓ Contains 5+ digits (e.g., 12345, 123456789)\n")
            f.write("  ✓ Can include / - symbols (e.g., WGN-12345, ABC/67890)\n")
            f.write("  ✗ Cannot include . , % symbols\n")
            f.write("  ℹ️  Numbers >7 digits are truncated in display (last 7 shown)\n")
        
        return report_path
    
    def export_video_pdf(self, video_id):
        """Export detailed PDF report with images for a video"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError:
            print("reportlab not installed. Install with: pip install reportlab")
            return None
        
        video = self.get_video_by_id(video_id)
        if not video:
            return None
        
        pdf_path = self.reports_path / f"{video_id}_report.pdf"
        
        # Create PDF
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title style
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#10b981'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        # Add title
        story.append(Paragraph(f"OCR Report: {video['video_name']}", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Add metadata
        metadata = [
            ['Scan Date:', video['timestamp'][:19]],
            ['Total Frames:', str(video['total_frames_processed'])],
            ['Text Detected:', str(video['total_text_detected'])],
            ['Frames with Text:', str(len(video['frames_with_text']))]
        ]
        
        t = Table(metadata, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#2e3140')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Text with Numbers section
        story.append(Paragraph("🔢 Text with Numbers", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        if video['text_with_numbers']:
            numbers_data = [['Text', 'Confidence', 'First Frame']]
            for item in video['text_with_numbers']:
                numbers_data.append([
                    item['text'],
                    f"{item['confidence']:.2%}",
                    str(item['first_seen_frame'])
                ])
            
            t = Table(numbers_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10b981')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(t)
        else:
            story.append(Paragraph("No text with numbers detected", styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Text Only section
        story.append(Paragraph("📝 Text Only (No Numbers)", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        if video['text_only']:
            text_data = [['Text', 'Confidence', 'First Frame']]
            for item in video['text_only']:
                text_data.append([
                    item['text'],
                    f"{item['confidence']:.2%}",
                    str(item['first_seen_frame'])
                ])
            
            t = Table(text_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(t)
        else:
            story.append(Paragraph("No text-only detected", styles['Normal']))
        
        story.append(PageBreak())
        
        # Frames with detected text
        story.append(Paragraph("🖼️ Frames with Detected Text", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        for frame_data in video['frames_with_text']:
            frame_path = self.get_frame_image_path(frame_data['frame_image'])
            
            if frame_path.exists():
                story.append(Paragraph(f"Frame {frame_data['frame_number']}", styles['Heading3']))
                
                # Add image
                try:
                    img = RLImage(str(frame_path), width=5*inch, height=3*inch)
                    story.append(img)
                except:
                    story.append(Paragraph(f"[Image: {frame_data['frame_image']}]", styles['Normal']))
                
                # Add detected text
                story.append(Spacer(1, 0.1*inch))
                texts = ", ".join([f"{t['text']} ({t['confidence']:.2%})" for t in frame_data['texts']])
                story.append(Paragraph(f"<b>Detected:</b> {texts}", styles['Normal']))
                story.append(Spacer(1, 0.3*inch))
        
        # Build PDF
        doc.build(story)
        return pdf_path
