Source: {source_path}

## Source content
{source_content}

## Current wiki index
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
      "title": "Natural Title Case name",
      "body": "full markdown body — use [[Wikilinks]] for cross-references",
      "description": "one-line description for the index"
    }
  ],
  "updated_pages": [
    {
      "title": "Exact existing page title",
      "body": "updated full markdown body",
      "timeline_tag": "[refined] OR [updated] OR [corrected]",
      "timeline_detail": "what changed and why"
    }
  ]
}
