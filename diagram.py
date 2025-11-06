
with tab7:
    st.markdown("""
    <style>
        .diagram-header {
            text-align: center;
            padding: 0.8rem;
            background: linear-gradient(90deg, #3b82f6, #9333ea);
            color: white;
            border-radius: 10px;
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1.2rem;
        }
        .diagram-box {
            background-color: #f9fafc;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        }
        .diagram-footer {
            text-align: center;
            margin-top: 1.5rem;
            font-size: 0.9rem;
            color: #6b7280;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="diagram-header">✨ Azure AI Diagram Generator</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; color:#4b5563; font-size:1.05rem;">
        Generate professional architecture visuals powered by Azure OpenAI and Mermaid.
    </div>
    """, unsafe_allow_html=True)

    # Choose diagram type (Dropdown instead of Radio)
    diagram_type = st.selectbox(
        "🧩 Choose diagram type:",
        [
            "ER Diagram",
            "Flowchart",
            "Sequence Diagram",
            "Class Diagram",
            "Architecture Diagram",
            "Component Diagram",
            "Deployment Diagram"
        ],
        index=0
    )

    # Diagram generation section
    st.markdown('<div class="diagram-box">', unsafe_allow_html=True)
    generate_button = st.button("🎨 Generate Diagram Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    def generate_mermaid(diagram_type):
        idx, texts, meta ,tokenized, vecs = load_index(index_path)
        seed_results = query(idx, texts, meta, "overview main modules data access business processes controllers", top_k=15)
        content = generate_mermaid_script(
            seed_results,
            diagram_type,
            parsed["nodes"],
            parsed.get("metrics"),
            parsed.get("business_processes"),
            power_mapping
        )
        print("Generated Mermaid Content:\n", content)
        match = re.search(r"```mermaid\n(.*?)```", content, re.DOTALL)
        return match.group(1).strip() if match else content.strip()

    if generate_button:
        with st.spinner("⚙️ Generating diagram..."):
            try:
                mermaid_code = generate_mermaid(diagram_type)
                print("Generated Mermaid Code:\n", mermaid_code)
                with tempfile.TemporaryDirectory() as tmpdir:
                    input_path = os.path.join(tmpdir, "diagram.mmd")
                    output_path = os.path.join(tmpdir, "diagram.png")
                    with open(input_path, "w", encoding="utf-8") as f:
                        f.write(mermaid_code)

                    # Mermaid CLI path
                    mmdc_path = r"C:\Users\srini\AppData\Roaming\npm\mmdc.cmd"
                    result = subprocess.run(
                        [mmdc_path, "-i", input_path, "-o", output_path],
                        capture_output=True, text=True, shell=True
                    )

                    if result.returncode != 0:
                        st.error(f"❌ Mermaid rendering failed:\n{result.stderr}")
                    else:
                        with open(output_path, "rb") as f:
                            img_data = f.read()
                            st.image(img_data, caption=f"{diagram_type} Preview", width=None)
                            st.download_button(
                                label="📥 Download Diagram (PNG)",
                                data=img_data,
                                file_name=f"{diagram_type.replace(' ', '_').lower()}.png",
                                mime="image/png",
                                use_container_width=True
                            )
            except Exception as e:
                st.error(f"Error generating diagram: {str(e)}")

    st.markdown('<div class="diagram-footer">Powered by Azure OpenAI + Mermaid CLI ⚡</div>', unsafe_allow_html=True)

# BRDGENERATOR.py
# mermaid diagrams

def generate_mermaid_script(retrieved: List[Dict[str, Any]], 
                    diagram_type: str,
                 nodes: List[Dict[str, Any]], 
                 metrics: Dict[str, Any] = None,
                 business_processes: List[Dict[str, Any]] = None,
                 power_platform_mapping: Dict[str, Any] = None,
                 ) -> str:
    """Generate comprehensive BRD with metrics and Power Platform focus"""
    try:
        graph_summary = summarize_graph_enhanced(nodes)
        context = _make_context_snippets(retrieved, max_chars=15000)
        
        # Format additional context
        metrics_summary = format_metrics_summary(metrics or {})
        processes_summary = format_business_processes(business_processes or [])
        mapping_summary = format_power_platform_mapping(power_platform_mapping or {})
        
        messages = [
            {"role": "system", "content": f'''You are a software architect with the details given below
              Generate valid Mermaid code for a {diagram_type.lower()}.
              
              for chart genreate mermaid code for user flow . Use appropriate mermaid syntax for the diagram type.

              generate the mermaid code only for the flow described below.
              **Describe what the user does from start to finish in the current system.**  
                Focus on real-world flow, screen sequence, and system reactions.

                ### Format:
                #### Example Table:
                | Step | User Action | System Response | Data Interaction | Integration | Validation | Output |
                |------|--------------|-----------------|------------------|--------------|-------------|---------|

                #### Example Narrative (Readable Flow)
                > **Persona:** Underwriter  
                > **Process:** Document Upload and Extraction  
                > **Goal:** Upload and extract data for verification  

                1. User logs into the web application using corporate credentials.  
                2. Navigates to the **Document Management** module.  
                3. Clicks **Upload File** and selects or captures a screenshot/document.  
                4. Clicks **Save**, triggering a backend API to store the file in blob storage.  
                5. System confirms upload and displays file metadata.  
                6. User opens the **View Uploaded Files** screen.  
                7. Selects a file and clicks **Verify** — triggers validation service.  
                8. Clicks **Extract** — backend parser extracts data into database tables.  
                9. User clicks **Download** to export processed data to Excel/PDF.  
                10. System logs the transaction and sends confirmation email.  

                **Observations:**  
                - Manual verification can be automated in Power Automate.  
                - Extraction currently synchronous — optimize via async Power Flow.  
                - File handling can migrate to Dataverse File column.  

                🟢 *Include a similar AS-IS flow for each key process or module.*
                ### Important:
              strictly return only the mermaid code without any explanation or markdown formatting.
                Only output code between ```mermaid``` fences, no explanation.

              '''},
            {"role": "user", "content": f"""
            GRAPH:
            {graph_summary}

            {metrics_summary}

            {processes_summary}

            {mapping_summary}

            CONTEXT SNIPPETS:
            {context}
            """}
        ]
        content = _chat(messages, temperature=0.1)
        return content
    except Exception as e:
        print(f"⚠️  Error preparing BRD context: {str(e)}")
        return "Error generating BRD."
    # Generate BRD content

