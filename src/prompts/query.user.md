## Question
{question}

## Current Wiki Index
{index_content}

## Relevant Wiki Pages
{pages_section}

## Instructions
- Synthesize a direct answer to the question using the wiki pages above.
- Use [[Wikilinks]] to cite pages inline (e.g. "[[LLM Wiki 패턴]]에 따르면...").
- If the pages do not contain enough information, say so clearly in the answer.
- Set save_analysis to true only when the answer synthesizes multiple pages non-trivially.
- If save_analysis is true, provide analysis_title and analysis_body as a full markdown page.
  The analysis body should be self-contained with proper [[Wikilinks]] cross-references.

Return JSON with this exact structure:
{
  "answer": "synthesized answer with [[Wikilink]] citations",
  "citations": ["Exact Page Title 1", "Exact Page Title 2"],
  "save_analysis": true or false,
  "analysis_title": "Title of the analysis page (only if save_analysis is true)",
  "analysis_body": "Full markdown body (only if save_analysis is true)"
}
