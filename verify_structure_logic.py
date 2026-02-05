
import re
import unittest

def normalize(text):
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

def post_process_structure(content: str, section_title: str) -> str:
    if not content or not section_title:
        return content

    section_title_norm = normalize(section_title)
    
    # Pattern to find h3 tags
    h3_pattern = re.compile(r'(<h3>.*?</h3>)', re.DOTALL | re.IGNORECASE)
    parts = h3_pattern.split(content)
    
    processed_parts = []
    valid_h3_indices = [] 
    
    for i, part in enumerate(parts):
        if h3_pattern.match(part):
            # Extract text content from H3
            h3_text = re.sub(r'<[^>]+>', '', part).strip()
            h3_norm = normalize(h3_text)
            
            # 1. Check for duplicate header
            # We strictly check if subsection header is effectively the same as section header
            if h3_norm == section_title_norm:
                 # It's a duplicate, we skip adding this part (remove the header)
                 # We do NOT add it to processed_parts
                 continue
            
            # Track location of valid H3s
            valid_h3_indices.append(len(processed_parts))
            processed_parts.append(part)
        else:
            processed_parts.append(part)
            
    # 2. Check for single subsection
    # We count how many H3s are in the *processed* parts (filtered dupes)
    if len(valid_h3_indices) == 1:
        idx = valid_h3_indices[0]
        h3_part = processed_parts[idx]
        
        # Extract content inside tags (removing <h3> wrapper)
        h3_content = re.sub(r'</?h3>', '', h3_part, flags=re.IGNORECASE).strip()
        processed_parts[idx] = f"<p><strong>{h3_content}</strong></p>"

    return "".join(processed_parts)


class TestStructureLogic(unittest.TestCase):
    def test_duplicate_header_removal(self):
        section_title = "The Evolution of Theory"
        content = """
        <p>Intro text.</p>
        <h3>The Evolution of Theory</h3>
        <p>More text.</p>
        """
        # Note: logic removes the H3 part entirely.
        
        result = post_process_structure(content, section_title)
        
        self.assertNotIn("<h3>The Evolution of Theory</h3>", result)
        self.assertIn("<p>Intro text.</p>", result)
        self.assertIn("<p>More text.</p>", result)

    def test_single_subsection_flattening(self):
        section_title = "Main Topic"
        content = """
        <p>Intro.</p>
        <h3>Unique Subtopic</h3>
        <p>Details.</p>
        """
        
        result = post_process_structure(content, section_title)
        
        self.assertNotIn("<h3>Unique Subtopic</h3>", result)
        self.assertIn("<p><strong>Unique Subtopic</strong></p>", result)
    
    def test_multiple_subsections_kept(self):
        section_title = "Main Topic"
        content = """
        <p>Intro.</p>
        <h3>Subtopic 1</h3>
        <p>Details 1.</p>
        <h3>Subtopic 2</h3>
        <p>Details 2.</p>
        """
        
        result = post_process_structure(content, section_title)
        
        self.assertIn("<h3>Subtopic 1</h3>", result)
        self.assertIn("<h3>Subtopic 2</h3>", result)

    def test_duplicate_and_single_interaction(self):
        # Case where we have 2 H3s, but one is a duplicate.
        # Result is 1 valid H3 -> should be flattened.
        section_title = "Main Topic"
        content = """
        <p>Intro.</p>
        <h3>Main Topic</h3>
        <p>Redundant intro.</p>
        <h3>Real Subtopic</h3>
        <p>Details.</p>
        """
        
        result = post_process_structure(content, section_title)
        
        # Main Topic header removed
        self.assertNotIn("<h3>Main Topic</h3>", result)
        # Real Subtopic flattend because it's the only one left
        self.assertNotIn("<h3>Real Subtopic</h3>", result)
        self.assertIn("<p><strong>Real Subtopic</strong></p>", result)

if __name__ == '__main__':
    unittest.main()
