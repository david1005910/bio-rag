"""Demo service for testing without API keys"""

import random
from datetime import datetime, timedelta


# Sample biomedical papers for demo mode
DEMO_PAPERS = [
    {
        "pmid": "38123456",
        "title": "Novel CRISPR-Cas9 Gene Therapy Approaches for Treating Hereditary Diseases",
        "abstract": "This study explores innovative applications of CRISPR-Cas9 technology in treating hereditary genetic disorders. We demonstrate successful gene correction in patient-derived cells and discuss the implications for clinical translation. Our findings suggest that CRISPR-based therapies could revolutionize treatment options for patients with genetic diseases.",
        "authors": ["Zhang L", "Smith JR", "Kim H"],
        "journal": "Nature Medicine",
        "pub_date": "2024-01-15",
    },
    {
        "pmid": "38234567",
        "title": "Machine Learning Models for Early Detection of Type 2 Diabetes",
        "abstract": "We developed and validated machine learning algorithms for predicting Type 2 diabetes risk using electronic health records. Our XGBoost model achieved 92% accuracy in identifying at-risk patients up to 5 years before clinical diagnosis. These tools could enable earlier intervention and improved patient outcomes.",
        "authors": ["Johnson M", "Lee S", "Patel R"],
        "journal": "Diabetes Care",
        "pub_date": "2024-02-20",
    },
    {
        "pmid": "38345678",
        "title": "CAR-T Cell Therapy in Solid Tumors: Overcoming the Tumor Microenvironment",
        "abstract": "Chimeric antigen receptor T-cell therapy has shown remarkable success in hematological malignancies. This review discusses strategies to enhance CAR-T efficacy in solid tumors by modifying the tumor microenvironment. We highlight combination approaches with checkpoint inhibitors and cytokine support.",
        "authors": ["Brown A", "Wilson C", "Davis K"],
        "journal": "Cancer Research",
        "pub_date": "2024-03-10",
    },
    {
        "pmid": "38456789",
        "title": "mRNA Vaccine Development for Infectious Diseases Beyond COVID-19",
        "abstract": "Building on the success of COVID-19 mRNA vaccines, researchers are developing similar platforms for other infectious diseases. This paper reviews ongoing clinical trials for mRNA vaccines against influenza, HIV, and malaria. We discuss the advantages of mRNA technology and remaining challenges.",
        "authors": ["Garcia E", "Thompson S", "Chen W"],
        "journal": "Science Translational Medicine",
        "pub_date": "2024-04-05",
    },
    {
        "pmid": "38567890",
        "title": "Gut Microbiome and Neurological Disorders: The Gut-Brain Axis",
        "abstract": "Emerging research reveals strong connections between gut microbiota and brain function. We analyzed microbiome compositions in patients with Parkinson's disease and identified specific bacterial signatures. Our findings support therapeutic interventions targeting the gut-brain axis for neurological conditions.",
        "authors": ["Miller T", "Yamamoto K", "Roberts L"],
        "journal": "Nature Neuroscience",
        "pub_date": "2024-05-12",
    },
    {
        "pmid": "38678901",
        "title": "AI-Assisted Drug Discovery: From Target Identification to Clinical Trials",
        "abstract": "Artificial intelligence is transforming pharmaceutical research. We present a comprehensive framework for AI-driven drug discovery that reduced development timelines by 60%. Our deep learning models successfully predicted drug-target interactions and identified novel therapeutic candidates.",
        "authors": ["Anderson J", "Liu Q", "Peterson M"],
        "journal": "Drug Discovery Today",
        "pub_date": "2024-06-18",
    },
    {
        "pmid": "38789012",
        "title": "Stem Cell-Based Regenerative Medicine for Heart Failure",
        "abstract": "This clinical trial evaluated induced pluripotent stem cell-derived cardiomyocytes for treating heart failure. Patients receiving cell therapy showed significant improvement in cardiac function at 6-month follow-up. These results demonstrate the potential of regenerative approaches for cardiovascular disease.",
        "authors": ["Taylor R", "Nakamura H", "Williams E"],
        "journal": "Circulation",
        "pub_date": "2024-07-22",
    },
    {
        "pmid": "38890123",
        "title": "Precision Oncology: Biomarker-Guided Treatment Selection",
        "abstract": "Precision medicine enables tailored cancer treatment based on molecular profiling. We conducted a multi-center study comparing biomarker-guided therapy versus standard treatment. Patients in the precision oncology arm showed 40% improvement in progression-free survival.",
        "authors": ["Martinez C", "O'Brien P", "Singh A"],
        "journal": "Journal of Clinical Oncology",
        "pub_date": "2024-08-28",
    },
]

DEMO_ANSWERS = {
    "default": """Based on the available biomedical research, here's what I found:

The field of biomedical science has seen remarkable advances in recent years, particularly in areas such as gene therapy, precision medicine, and AI-assisted drug discovery. Key developments include:

1. **Gene Editing Technologies**: CRISPR-Cas9 continues to show promise for treating hereditary diseases through precise gene correction.

2. **Personalized Medicine**: Machine learning models are enabling earlier disease detection and more targeted treatment approaches.

3. **Immunotherapy Advances**: CAR-T cell therapy and other immunotherapies are expanding beyond blood cancers to solid tumors.

4. **Vaccine Technology**: mRNA vaccine platforms developed for COVID-19 are being adapted for other infectious diseases.

These advances represent significant progress toward more effective and personalized healthcare solutions.""",

    "diabetes": """Based on current research on Type 2 diabetes treatments:

**Latest Treatment Approaches:**

1. **GLP-1 Receptor Agonists**: Drugs like semaglutide (Ozempic) show significant benefits for both glycemic control and weight management.

2. **SGLT2 Inhibitors**: These medications not only lower blood sugar but also provide cardiovascular and renal protection.

3. **Early Detection**: Machine learning models can now predict diabetes risk up to 5 years before clinical diagnosis, enabling preventive interventions.

4. **Personalized Treatment**: Precision medicine approaches are helping identify which patients will respond best to specific treatments.

5. **Lifestyle Integration**: Digital health tools and continuous glucose monitors are improving patient self-management.

**Citations from research indicate that combination therapies targeting multiple pathways show the most promise for comprehensive disease management.**""",

    "crispr": """CRISPR-Cas9 gene editing works through a remarkably precise mechanism:

**How CRISPR Works:**

1. **Guide RNA (gRNA)**: A short RNA sequence is designed to match the target DNA sequence in the genome.

2. **Cas9 Protein**: This molecular "scissors" is directed to the target site by the guide RNA.

3. **DNA Cutting**: Cas9 creates a double-strand break at the precise location.

4. **Repair Mechanisms**: The cell's natural repair machinery then fixes the break, allowing for:
   - Gene knockout (disabling a gene)
   - Gene correction (fixing mutations)
   - Gene insertion (adding new genetic material)

**Current Applications:**
- Treating hereditary diseases like sickle cell disease
- Cancer immunotherapy enhancement
- Agricultural improvements
- Research tool for understanding gene function

**Recent advances show successful clinical applications in treating genetic blood disorders with high efficacy and safety profiles.**""",

    "car-t": """CAR-T cell therapy represents a revolutionary approach to cancer treatment:

**Mechanism of CAR-T Therapy:**

1. **T-Cell Collection**: Immune cells are extracted from the patient's blood.

2. **Genetic Engineering**: T-cells are modified to express Chimeric Antigen Receptors (CARs) that recognize cancer cells.

3. **Cell Expansion**: Modified cells are multiplied in the laboratory.

4. **Infusion**: Engineered cells are returned to the patient to attack cancer.

**Current Status:**
- FDA-approved for certain blood cancers (leukemias, lymphomas)
- Active research for solid tumors
- Combination approaches with checkpoint inhibitors showing promise

**Challenges Being Addressed:**
- Tumor microenvironment suppression
- CAR-T cell persistence
- Managing cytokine release syndrome

**Research indicates that overcoming the immunosuppressive tumor microenvironment is key to expanding CAR-T success to solid tumors.**""",
}


def get_demo_search_results(query: str, limit: int = 10) -> dict:
    """Get demo search results based on query"""
    query_lower = query.lower()

    # Filter papers based on query relevance
    scored_papers = []
    for paper in DEMO_PAPERS:
        score = 0
        search_text = f"{paper['title']} {paper['abstract']}".lower()

        # Simple keyword matching for demo
        for word in query_lower.split():
            if word in search_text:
                score += 1

        if score > 0 or len(scored_papers) < 3:  # Always return at least 3 papers
            scored_papers.append((score, paper))

    # Sort by score and take top results
    scored_papers.sort(key=lambda x: x[0], reverse=True)
    results = [p[1] for p in scored_papers[:limit]]

    # If no matches, return random papers
    if not results:
        results = random.sample(DEMO_PAPERS, min(limit, len(DEMO_PAPERS)))

    return {
        "results": [
            {
                "pmid": p["pmid"],
                "title": p["title"],
                "authors": p["authors"],
                "journal": p["journal"],
                "publication_date": p["pub_date"],
                "abstract": p["abstract"][:300],
                "relevance_score": round(random.uniform(70, 95), 1),
            }
            for p in results
        ],
        "total": len(results),
        "query_time_ms": random.randint(50, 200),
    }


def get_demo_chat_response(query: str) -> dict:
    """Get demo chat response based on query"""
    query_lower = query.lower()

    # Select appropriate answer based on keywords
    if "diabetes" in query_lower or "glucose" in query_lower or "insulin" in query_lower:
        answer = DEMO_ANSWERS["diabetes"]
    elif "crispr" in query_lower or "gene edit" in query_lower or "cas9" in query_lower:
        answer = DEMO_ANSWERS["crispr"]
    elif "car-t" in query_lower or "car t" in query_lower or "immunotherapy" in query_lower:
        answer = DEMO_ANSWERS["car-t"]
    else:
        answer = DEMO_ANSWERS["default"]

    # Select random citations
    citations = random.sample(DEMO_PAPERS, min(3, len(DEMO_PAPERS)))

    return {
        "answer": answer,
        "citations": [
            {
                "pmid": p["pmid"],
                "title": p["title"],
                "relevance_score": round(random.uniform(0.75, 0.95), 2),
                "snippet": p["abstract"][:150] + "...",
            }
            for p in citations
        ],
    }
