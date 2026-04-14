"""
OCR Database Manager
Handles persistent storage of OCR results with frame images
"""

import json
import os
from datetime import datetime
from pathlib import Path
import cv2
import re


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
            "total_frames_processed": 0,
            "total_text_detected": 0
        }
        
        self.index["videos"].append(video_entry)
        self.index["total_scans"] += 1
        self.save_index()
        
        return video_id
    
    def has_numbers(self, text):
        """Check if text contains numbers"""
        return bool(re.search(r'\d', text))
    
    def filter_text(self, text):
        """Filter text - must be 3+ characters"""
        return len(text.strip()) >= 3
    
    def add_ocr_result(self, video_id, frame_number, frame_image, text_results):
        """Add OCR results for a frame"""
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
            
            # Categorize
            if self.has_numbers(text):
                if text not in [item['text'] for item in video_entry['text_with_numbers']]:
                    video_entry['text_with_numbers'].append({
                        'text': text,
                        'confidence': confidence,
                        'first_seen_frame': frame_number
                    })
            else:
                if text not in [item['text'] for item in video_entry['text_only']]:
                    video_entry['text_only'].append({
                        'text': text,
                        'confidence': confidence,
                        'first_seen_frame': frame_number
                    })
        
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
    
    def export_video_report(self, video_id):
        """Export detailed report for a video"""
        video = self.get_video_by_id(video_id)
        if not video:
            return None
        
        report_path = self.reports_path / f"{video_id}_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"OCR REPORT: {video['video_name']}\n")
            f.write(f"Scan Date: {video['timestamp']}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Frames Processed: {video['total_frames_processed']}\n")
            f.write(f"Total Text Detected: {video['total_text_detected']}\n")
            f.write(f"Frames with Text: {len(video['frames_with_text'])}\n\n")
            
            f.write("="*80 + "\n")
            f.write("TEXT WITH NUMBERS\n")
            f.write("="*80 + "\n")
            for item in video['text_with_numbers']:
                f.write(f"  • {item['text']} (confidence: {item['confidence']:.2f}, frame: {item['first_seen_frame']})\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("TEXT ONLY (NO NUMBERS)\n")
            f.write("="*80 + "\n")
            for item in video['text_only']:
                f.write(f"  • {item['text']} (confidence: {item['confidence']:.2f}, frame: {item['first_seen_frame']})\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("FRAME-BY-FRAME DETAILS\n")
            f.write("="*80 + "\n")
            for frame_data in video['frames_with_text']:
                f.write(f"\nFrame {frame_data['frame_number']}:\n")
                f.write(f"  Image: {frame_data['frame_image']}\n")
                f.write(f"  Detected Text:\n")
                for text in frame_data['texts']:
                    f.write(f"    - {text['text']} (confidence: {text['confidence']:.2f})\n")
        
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
