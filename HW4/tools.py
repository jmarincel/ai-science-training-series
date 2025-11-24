from typing import Any, Dict, Literal
import os
from langchain_core.tools import tool

@tool
def lookup_on_arxiv(query: str) -> str:
    """Find papers on the arxiv matching search criteria. 

    Parameters
    ----------
    query : str
        The search query. 

    Returns
    -------
    str
        A list of URLs pointing to papers on the arxiv. 
    """
    import arxiv
    
    # Construct the default API client.
    client = arxiv.Client()
    
    # Search for the 3 most relevant articles matching the query.
    search = arxiv.Search(
        query=query,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    # `results` is a list. 
    results = [(r.title, r.entry_id) for r in client.results(search)]

    return results

@tool
def download_arxiv_paper(arxiv_entry: tuple) -> int: 
    """Download pdf file of a paper from the arxiv. 

    Parameters
    ----------
    arxiv_entry : tuple
        A (title, URL) pair corresponding to a paper on the arxiv. 

    Returns
    -------
    0
    """
    import requests

    #extract title and URL
    title, hyperlink = arxiv_entry
    #construct pdf URL
    pdf_link = hyperlink.replace('abs', 'pdf')
    #construct filename
    filename = title + '.pdf'
    #fetch and save content
    response = requests.get(pdf_link)
    with open(filename, mode="wb") as file:
        file.write(response.content)

    return 0