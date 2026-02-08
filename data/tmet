graph TD
    %% Nodes
    Root[("QR-Rhetoric Record<br/>(JSON Object)")]
    
    %% Main Layers
    Meta[/"1. Metadata Layer"/]
    Struct[/"2. Structural Layer"/]
    Sem[/"3. Semantic Layer"/]
    Prag[/"4. Pragmatic Layer"/]

    %% Metadata Children
    M1[Record ID & Verse Ref]
    M2[Annotator Provenance]
    M3[Confidence Score]

    %% Structural Children
    S1[Tenor / Target Entity]
    S2[Vehicle / Source Entity]
    S3[Ground / Basis]
    S4[Marker / Particle]

    %% Semantic Children
    Sem1[Source Domain Ontology]
    Sem2[Target Domain Ontology]
    Sem3[Entailment Structure]

    %% Pragmatic Children
    P1[Speech Act Taxonomy]
    P2[Sensory Modality]
    P3[Rhetorical Intensity]

    %% Connections
    Root --> Meta
    Root --> Struct
    Root --> Sem
    Root --> Prag

    Meta --- M1
    Meta --- M2
    Meta --- M3

    Struct --- S1
    Struct --- S2
    Struct --- S3
    Struct --- S4

    Sem --- Sem1
    Sem --- Sem2
    Sem --- Sem3

    Prag --- P1
    Prag --- P2
    Prag --- P3

    %% Styling
    style Root fill:#f9f,stroke:#333,stroke-width:4px
    style Meta fill:#e1f5fe,stroke:#01579b
    style Struct fill:#e8f5e9,stroke:#1b5e20
    style Sem fill:#fff3e0,stroke:#e65100
    style Prag fill:#f3e5f5,stroke:#4a148c
