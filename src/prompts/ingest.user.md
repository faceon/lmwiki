## Source Path
{source_path}

## Source Content
{source_content}

## Current Wiki Index
{index_content}
{related_section}

## Instructions
- Pages with similarity ≥ 60%: put in updated_pages (do NOT create a new page for the same topic)
- Pages with similarity < 40% and no related page: put in new_pages
- When in doubt, prefer updated_pages over new_pages

Return JSON with this exact structure:
{
  "new_pages": [
    {
      "type": "concept OR entity",
      "title": "Title Case name — entities get @ prefix (e.g. @Google), concepts have none (e.g. LLM Wiki)",
      "body": "full markdown body — use [[Wikilinks]] for cross-references",
      "description": "One complete sentence, maximum 100 characters"
    }
  ],
  "updated_pages": [
    {
      "title": "Exact existing page title",
      "body": "updated full markdown body",
      "description": "One complete sentence, maximum 100 characters",
      "timeline_tag": "[refined] OR [updated] OR [corrected]",
      "timeline_detail": "what changed and why"
    }
  ]
}
