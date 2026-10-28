# Document Pipeline Overview

This pipeline turns a directory of mixed documents into a searchable index.

## Stages

The pipeline runs a sequence of stages. First it detects the file type, then it
extracts the content with the right parser. After extraction the text is
normalized and validated.

## Chunking

Chunking splits long documents into overlapping pieces so that retrieval can
return focused passages. Each chunk keeps a reference to its source heading and,
for PDFs, the page it came from.

| Stage      | Purpose                        |
|------------|--------------------------------|
| Extract    | Pull text out of the file      |
| Normalize  | Clean and canonicalize text    |
| Chunk      | Split into retrievable units   |
| Embed      | Turn text into vectors         |
