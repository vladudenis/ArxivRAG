"""
Section detection for academic papers.
Identifies sections like Introduction, Method, Experiments, Conclusion, etc.
"""
import re
from typing import List, Dict, Tuple

class SectionDetector:
    """Detects sections in academic paper text."""
    
    # Common section headers in academic papers
    SECTION_PATTERNS = [
        # Standard sections
        (r'^\s*(?:1\.?\s*)?(?:Introduction|INTRODUCTION)\s*$', 'Introduction'),
        (r'^\s*(?:2\.?\s*)?(?:Related\s+Work|RELATED\s+WORK|Background|BACKGROUND)\s*$', 'Related Work'),
        (r'^\s*(?:3\.?\s*)?(?:Method|METHOD|Methodology|METHODOLOGY|Approach|APPROACH)\s*$', 'Method'),
        (r'^\s*(?:4\.?\s*)?(?:Experiments?|EXPERIMENTS?|Evaluation|EVALUATION|Results?|RESULTS?)\s*$', 'Experiments'),
        (r'^\s*(?:5\.?\s*)?(?:Discussion|DISCUSSION)\s*$', 'Discussion'),
        (r'^\s*(?:6\.?\s*)?(?:Conclusion|CONCLUSION|Conclusions?|CONCLUSIONS?)\s*$', 'Conclusion'),
        
        # Variations
        (r'^\s*(?:[0-9]+\.?\s*)?(?:Preliminaries|PRELIMINARIES)\s*$', 'Preliminaries'),
        (r'^\s*(?:[0-9]+\.?\s*)?(?:Problem\s+Formulation|PROBLEM\s+FORMULATION)\s*$', 'Problem Formulation'),
        (r'^\s*(?:[0-9]+\.?\s*)?(?:Architecture|ARCHITECTURE)\s*$', 'Architecture'),
        (r'^\s*(?:[0-9]+\.?\s*)?(?:Implementation|IMPLEMENTATION)\s*$', 'Implementation'),
        (r'^\s*(?:[0-9]+\.?\s*)?(?:Analysis|ANALYSIS)\s*$', 'Analysis'),
        (r'^\s*(?:[0-9]+\.?\s*)?(?:Future\s+Work|FUTURE\s+WORK)\s*$', 'Future Work'),
        
        # Abstract and References (usually at start/end)
        (r'^\s*(?:Abstract|ABSTRACT)\s*$', 'Abstract'),
        (r'^\s*(?:References?|REFERENCES?|Bibliography|BIBLIOGRAPHY)\s*$', 'References'),
        
        # Numbered sections (fallback)
        (r'^\s*([0-9]+)\.?\s+(.+?)\s*$', None),  # Will extract number and title
    ]
    
    def __init__(self):
        """Initialize section detector."""
        self.compiled_patterns = [
            (re.compile(pattern, re.MULTILINE | re.IGNORECASE), name)
            for pattern, name in self.SECTION_PATTERNS[:-1]  # Exclude numbered pattern
        ]
        # Compile numbered pattern separately
        self.numbered_pattern = re.compile(self.SECTION_PATTERNS[-1][0], re.MULTILINE | re.IGNORECASE)
    
    def detect_sections(self, text: str) -> List[Tuple[int, str, str]]:
        """
        Detect sections in paper text.
        
        Args:
            text: Full paper text
            
        Returns:
            List of tuples: (start_index, section_name, section_text)
            Sections are ordered by appearance in text.
        """
        sections = []
        lines = text.split('\n')
        
        # Track current section
        current_section = 'Unknown'
        current_start = 0
        current_text = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this line is a section header
            section_name = self._match_section_header(line)
            
            if section_name:
                # Save previous section if it has content
                if current_text:
                    section_text = '\n'.join(current_text).strip()
                    if section_text:
                        sections.append((current_start, current_section, section_text))
                
                # Start new section
                current_section = section_name
                current_start = i
                current_text = []
            else:
                # Add line to current section
                current_text.append(lines[i])
            
            i += 1
        
        # Add final section
        if current_text:
            section_text = '\n'.join(current_text).strip()
            if section_text:
                sections.append((current_start, current_section, section_text))
        
        # If no sections detected, treat entire text as one section
        if not sections:
            sections.append((0, 'Full Text', text))
        
        return sections
    
    def _match_section_header(self, line: str) -> str:
        """
        Check if a line matches a section header pattern.
        
        Args:
            line: Line to check
            
        Returns:
            Section name if match found, None otherwise
        """
        # Check compiled patterns
        for pattern, name in self.compiled_patterns:
            if pattern.match(line):
                return name
        
        # Check numbered pattern (extract meaningful title)
        match = self.numbered_pattern.match(line)
        if match and len(match.groups()) >= 2:
            # Extract title and check if it's meaningful
            title = match.group(2).strip()
            # Heuristic: if title is short and capitalized, likely a section
            if title and len(title) < 50 and title[0].isupper():
                return title
        
        return None
    
    def get_section_text(self, text: str, section_name: str) -> str:
        """
        Extract text for a specific section.
        
        Args:
            text: Full paper text
            section_name: Name of section to extract
            
        Returns:
            Section text, or empty string if not found
        """
        sections = self.detect_sections(text)
        for _, name, section_text in sections:
            if name.lower() == section_name.lower():
                return section_text
        return ""
    
    def get_all_section_names(self, text: str) -> List[str]:
        """
        Get list of all section names in the paper.
        
        Args:
            text: Full paper text
            
        Returns:
            List of section names
        """
        sections = self.detect_sections(text)
        return [name for _, name, _ in sections]

