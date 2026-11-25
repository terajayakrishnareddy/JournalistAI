# JournalistAI: An Edge-Cloud Hybrid Architecture for Automated Speech Transcription, Verbatim Quotation Extraction, and Fact-Checking Verification in Journalism Workflows

Ayyappa Reddy¹
¹ Department of Computer Science and Engineering, Lovely Professional University, Phagwara, Punjab, India
ayyappareddy49777@gmail.com

Abstract
Investigative journalism relies heavily on spoken discourse from confidential interviews, press conferences, and whistleblower recordings. However, conventional manual transcription is prohibitively labor-intensive, while commercial cloud-only automated speech recognition (ASR) engines introduce source-protection vulnerabilities, lack deterministic safeguards against generative quotation hallucination, and fail to isolate verifiable claims from editorial conjecture. In this paper, we propose and evaluate JournalistAI, an edge-cloud hybrid platform tailored for newsroom investigative workflows. The architecture incorporates an on-device, 8-bit quantized Transformer ASR engine (OpenAI Whisper large-v3 executed via CTranslate2) combined with voice activity detection (VAD), chunked multi-core parallel inference, and pyannote.audio speaker diarization. Downstream natural language understanding (NLU) is performed by Google Gemini 2.5 Pro with an automatic local rule-based heuristic fallback. We introduce two programmatic invariants: (i) an exact substring verification constraint that discards non-verbatim candidate quotations at the code level, eliminating generative hallucinations, and (ii) an epistemic non-assertion gate that prevents generative models from arbitrating factual truth, strictly funneling detected claims into an investigative human verification ledger with an initial status of needs_verification. In empirical benchmarks across 14.8 hours of heterogeneous journalistic recordings, JournalistAI achieves a Real-Time Factor (RTF) of 0.14x in parallel turbo mode (a 4.1x throughput improvement over single-threaded execution) while maintaining 100% quotation fidelity and zero audio data egress.

Keywords: Automated Speech Recognition · Verbatim Extraction · Fact-Checking Pipeline · Whisper Transformer · Gemini LLM · Investigative Journalism · Privacy-Preserving NLP

1 Introduction
The contemporary newsroom operates under a dual constraint: the imperative for rapid investigative publication and the non-negotiable obligation of factual rigor and source confidentiality. Spoken discourse—encompassing confidential whistleblower interviews, background briefings, public hearings, and investigative field recordings—constitutes the foundational evidentiary substrate of investigative reporting. Nonetheless, transforming raw multi-hour audio streams into structured, verifiable news artifacts remains a critical operational bottleneck.

Manual transcription typically demands between four to six hours per single hour of recorded conversation. While commercial Software-as-a-Service (SaaS) cloud transcription platforms have proliferated, their adoption in investigative newsrooms introduces severe operational and ethical deficiencies:
1. Source Privacy and Data Sovereignty: Uploading proprietary audio recordings of sensitive sources to external third-party multi-tenant cloud APIs directly breaches confidentiality agreements and endangers whistleblowers.
2. Generative Quotation Hallucination: Contemporary generative large language models (LLMs) used for summarization frequently paraphrase or introduce subtle semantic distortions into candidate quotations. In investigative journalism, a non-verbatim quotation misattributed to a public official or source constitutes grounds for legal defamation.
3. Epistemic Overreach in Fact-Checking: General-purpose artificial intelligence models exhibit well-documented overconfidence when evaluating truth claims, hallucinating citations or asserting correctness on topics outside their temporal or factual knowledge horizon.

To resolve this trilemma, this paper introduces JournalistAI, an open-source, hybrid edge-cloud transcription and insight verification platform engineered specifically for journalism workflows. JournalistAI processes raw audio 100% locally on user hardware using an 8-bit integer quantized implementation of OpenAI's Whisper large-v3 model via CTranslate2. Only verified, redacted text transcripts are forwarded to downstream LLM reasoning engines (Google Gemini 2.5 Pro or local offline rule-based heuristic analyzers).

The main contributions of this paper are:
- Zero-Egress Acoustic Pipeline with Parallelized Throughput: We architect an on-device signal processing and ASR pipeline using ffmpeg, ffprobe, Silero VAD, and multi-core CTranslate2 Whisper execution that achieves an empirical Real-Time Factor (RTF) of 0.14x without releasing raw acoustic data beyond the newsroom perimeter.
- Deterministic Verbatim Quotation Guarantee: We formulate and implement a programmatic verification invariant that checks candidate quotes against the primary ASR token stream prior to database persistence, establishing an absolute theoretical and practical bound against quotation hallucination.
- Epistemic Fact-Checking Human-in-the-Loop Architecture: We design a journalistic claim extraction protocol that classifies factual propositions based on verifiable predicate-argument structures while strictly enforcing an immutable initial status of needs_verification, precluding algorithmic misattribution of truth.

The remainder of this paper is organized as follows: Section 2 synthesizes related literature in speech recognition and computational journalism. Section 3 outlines the formal problem formulation, mathematical notations, and research objectives. Section 4 presents the proposed edge-cloud system architecture, algorithms, and implementation stack. Section 5 describes the experimental design, dataset characteristics, and baseline metrics. Section 6 presents quantitative evaluation results and discussion. Section 7 assesses threats to validity and limitations. Section 8 concludes the paper and outlines future research trajectories.

2 Related Work
Modern speech processing transitioned decisively from Gaussian Mixture Model–Hidden Markov Models (GMM-HMMs) to end-to-end neural Transformer architectures. Radford et al. demonstrated that training weak-supervised models on 680,000 hours of multilingual audio yields robust cross-domain generalization without fine-tuning [1]. However, deploying large autoregressive Transformer decoders (such as Whisper large-v3 with 1.55 billion parameters) on consumer hardware imposes substantial computational and memory burdens. Klein et al. introduced CTranslate2, which applies weight quantization (int8) and operator fusion to achieve accelerated execution on commodity CPUs and GPUs [2]. While general transcription has improved, applying ASR directly to newsroom workflows requires addressing audio preprocessing, variable sampling rates, and strict adherence to verbatim accuracy.

Information extraction from transcripts has historically relied on Named Entity Recognition (NER) and syntactic dependency parsing. In journalistic domains, extracting pull-quotes requires distinguishing significant assertions from filler discourse. Recent approaches leverage instruction-tuned Large Language Models [3]. However, generative LLMs are prone to unfaithful text generation and summarization hallucinations, where models substitute near-synonyms or alter sentence polarity [4]. In journalism, paraphrasing a quote compromises attribution integrity. Prior literature has explored post-hoc hallucination detection [5], but few systems enforce deterministic token-matching filters at the ingestion interface.

Computational fact-checking traditionally spans three stages: claim detection, evidence retrieval, and claim verification [6]. Automated claim detection identifies check-worthy factual assertions—distinguishing empirical propositions (e.g., budgetary figures, legislative votes) from personal opinions or normative judgments [7]. While commercial platforms attempt end-to-end automated veracity scoring, studies demonstrate that automated truth verdicts risk amplifying misinformation due to temporal staleness and ungrounded reasoning [8]. Current ethical journalism guidelines dictate that computation should accelerate claim identification and evidence discovery, leaving final verification to human editorial judgment.

3 Problem Formulation / Research Objectives
Let an input media recording be represented as a continuous time-domain signal X in R^T with duration D seconds, accompanied by metadata M. The transcription objective is to map signal X to an ordered sequence of N timestamped transcript segments:
S = {(s_i, t_i^start, t_i^end, p_i, w_i)}_{i=1}^N
where s_i is the normalized text sequence, [t_i^start, t_i^end] defines the temporal boundary, p_i represents speaker attribution, and w_i = {(v_ij, tau_ij^start, tau_ij^end)} denotes word-level token alignments.

The downstream understanding task requires mapping the reconstructed aggregate transcript T = bigoplus_{i=1}^N s_i into a three-tuple:
Phi(T) -> (Q, C, K)
where:
- Q = {q_k | q_k is a substring of T} is the set of verbatim candidate quotations.
- C = {(c_m, kappa_m, sigma_m)}_{m=1}^M is the set of factual claims, where c_m is the proposition, kappa_m is the category, and sigma_m in {needs_verification} is the initial verification status.
- K = (Y, Theta) represents document-level synthesis comprising abstractive summary Y and key topical entities Theta.

3.1 Research Questions or Objectives
The system design and empirical evaluations are governed by three explicit research questions and objectives:
- RQ1 / Objective 1 (Acoustic Efficiency and Privacy): Can an edge-deployed, 8-bit quantized Transformer ASR model achieve a Real-Time Factor (RTF < 0.25) on standard multi-core CPU architectures while ensuring zero leakage of voice signals?
- RQ2 / Objective 2 (Verbatim Reliability): How can generative LLM extraction pipelines be constrained to ensure mathematical zero-hallucination in published candidate quotations?
- RQ3 / Objective 3 (Claim Isolation): Does heuristic and semantic claim detection reliably isolate verifiable empirical assertions from unstructured interview discourse without automated truth-verdict bias?

4 Proposed Methodology / System Model
The JournalistAI system is partitioned into two distinct security and execution rings: the Edge Privacy Ring and the Constrained Cloud Reasoning Ring.

4.1 Overall Framework
The architectural workflow separates local audio decoding from text-only natural language extraction:

```
+------------------------------------------------------------------------------------+
|                               EDGE PRIVACY RING                                    |
|                                                                                    |
|  [Raw Audio/Video File]                                                            |
|          |                                                                         |
|          v                                                                         |
|  +--------------------+      ffprobe Container Validation                          |
|  | Media Validator    | ---> Duration, Bitrate, Codec Whitelist                    |
|  +--------------------+                                                            |
|          |                                                                         |
|          v                                                                         |
|  +--------------------+      ffmpeg 16kHz Mono PCM Extraction                      |
|  | Audio Processor    | ---> -vn -ac 1 -ar 16000 -acodec pcm_s16le                 |
|  +--------------------+                                                            |
|          |                                                                         |
|          v                                                                         |
|  +--------------------+      Chunked Multi-Core Parallel Execution                 |
|  | faster-whisper     | ---> Silero VAD + CTranslate2 int8 Quantization            |
|  | large-v3 ASR       | ---> Word Timestamps & Log-Probability Extraction          |
|  +--------------------+                                                            |
|          |                                                                         |
|          v                                                                         |
|  +--------------------+      Optional Diarization Clustering                       |
|  | pyannote.audio 3.1 | ---> Speaker Diarization [spk_0, spk_1, ...]               |
|  +--------------------+                                                            |
+------------------------------------------------------------------------------------+
                                   | Verified Transcript Text Only (Zero Audio Egress)
                                   v
+------------------------------------------------------------------------------------+
|                         CONSTRAINED CLOUD / FALLBACK RING                          |
|                                                                                    |
|  +--------------------+      JSON Schema Strict Parsing                            |
|  | Google Gemini 2.5  | ---> Candidate Quotes, Claims, Topics, Summary             |
|  | Pro / Local Engine |                                                            |
|  +--------------------+                                                            |
|          |                                                                         |
|          v                                                                         |
|  +--------------------+      Verbatim Substring Matching Gate                      |
|  | Quote Validator    | ---> If q_k not in Transcript: DROP (Zero Hallucination)   |
|  +--------------------+                                                            |
|          |                                                                         |
|          v                                                                         |
|  +--------------------+      Epistemic Safeguard Filter                            |
|  | Claim Gate         | ---> Hard-lock Status = "needs_verification"               |
|  +--------------------+                                                            |
|          |                                                                         |
|          v                                                                         |
|  +--------------------+      Relational Persistence & Newsroom Exports             |
|  | SQLite / Storage   | ---> Web Studio UI, SRT/VTT, TXT/CSV/JSON, Dossier         |
|  +--------------------+                                                            |
+------------------------------------------------------------------------------------+
```
Fig. 1. Architectural schematic of the JournalistAI platform, distinguishing local edge signal processing from constrained natural language understanding.

4.2 Algorithm / Procedure
The transcription and extraction procedure operates according to three computational algorithms:

1. Parallelized Turbo ASR Transcription:
Audio files exceeding duration threshold tau = 60 s are partitioned into K uniform acoustic temporal chunks:
C_k = [(k - 1) * (D / K), k * (D / K)], for k in {1, ..., K}
Each chunk is dispatched to a thread-safe faster-whisper worker instance operating within a bounded thread pool. Each worker runs Voice Activity Detection (VAD) filtering to strip non-speech intervals. The decoding confidence of each segment s_i is computed directly from the mean token log-probabilities:
Confidence(s_i) = exp( (1 / |s_i|) * sum_{j=1}^{|s_i|} log P(w_ij | w_i,<j, X) )
Segment timestamps from chunk C_k are shifted by (k - 1) * (D / K) prior to sequential concatenation.

2. Verbatim Quote Validation Procedure:
Candidate quotations Q_raw returned by the LLM are evaluated against transcript T through deterministic normalized substring search:
- Initialize Q_verified as an empty collection.
- Clean and normalize transcript T: T_clean = NormalizeWhitespace(T.toLowerCase()).
- For each candidate quote q in Q_raw:
  - Normalize candidate text: q_clean = NormalizeWhitespace(q.text.toLowerCase()).
  - If q_clean is a substring of T_clean, append q to Q_verified.
  - Else, drop candidate q and log an unverified hallucination rejection.
- Return Q_verified.

3. Epistemic Human-in-the-Loop Claim Governance:
Extracted claims C undergo programmatic schema sanitization. Regardless of any external context provided by the LLM, the system enforces:
For all c in C: status(c) <- "needs_verification"
The human reporter is provided interactive tools to mark claims as supported, refuted, or unverified based on live retrieval of external journalistic sources.

4.3 Implementation Details
The backend is implemented in Python 3.11 using the FastAPI asynchronous web framework. Media validation relies on ffprobe to verify file headers, stream counts, duration constraints (<= 10,800 s), and payload sizes (<= 2 GB). Audio extraction outputs canonical 16 kHz 16-bit mono PCM streams via ffmpeg. Database persistence utilizes SQLAlchemy ORM backed by SQLite with write-ahead logging (WAL) enabled. The web client is built with vanilla ECMAScript 2026, responsive CSS custom properties, and SVG visualization, providing scrubber-synchronized audio playback, interactive word tracking, and one-click export to SubRip (.srt), WebVTT (.vtt), Markdown, and structured JSON formats.

5 Experimental Setup / Evaluation Design

5.1 Dataset / Data Sources
Evaluation was conducted on a curated benchmark of 25 multi-genre spoken audio sessions totaling 14.8 hours:
- Political Interviews & Press Conferences (10 sessions): High-density technical vocabulary, overlapping dialogue, ambient acoustic noise.
- Investigative Field Recordings (8 sessions): Low signal-to-noise ratio (SNR), reverberant environments, covert mic placements.
- Academic & Policy Briefings (7 sessions): Monologue lectures with high syntactic complexity and formal statistical claims.
Preprocessing: All streams are transcoded to uncompressed 16 kHz mono WAV via ffmpeg. Whitelist validation enforces codec legitimacy (rejecting malicious container exploits).
Ethical and Privacy Considerations: Audio samples were collected from public journalistic archives and simulated investigative briefings under informed consent. No proprietary whistleblower audio was exposed to external networks.

5.2 Baselines and Evaluation Metrics
We compare JournalistAI across three operational baselines:
1. Whisper large-v3 (Sequential): Standard single-threaded execution of Whisper large-v3 without VAD or chunking.
2. Local Fallback Heuristic Engine: Offline rule-based regex extraction and frequency-based topic modeling.
3. Commercial Cloud ASR + Generative LLM: Third-party commercial speech API paired with unconstrained prompt-based quotation extraction.

Evaluation Metrics:
- Real-Time Factor (RTF): Ratio of wall-clock processing time to audio duration. Lower values indicate higher throughput (RTF < 1.0 represents faster-than-real-time).
- Quotation Hallucination Rate (QHR, %): Percentage of candidate quotations whose exact text does not appear verbatim in the primary ASR transcript.
- Quotation Fidelity (%): Percentage of extracted quotes certified as identical verbatim substrings of the source transcript.
- Claim Extraction Yield: Number of verifiable empirical assertions detected per hour of audio.
- Source Audio Egress: Degree of raw audio exposure to external networks (Zero / Full).

6 Results and Discussion

6.1 Quantitative Results
Table 1 summarizes the performance comparison between the evaluated baselines and the proposed JournalistAI system.

| Method | Real-Time Factor (RTF) | Quotation Fidelity (%) | Quotation Hallucination Rate (QHR, %) | Claim Yield (per hour) | Audio Data Egress |
|---|---|---|---|---|---|
| Whisper large-v3 (Sequential) | 0.58 | N/A (ASR only) | N/A | N/A | Zero (Local) |
| Local Fallback Engine | 0.15 | 100.0% | 0.0% | 4.8 | Zero (Local) |
| Commercial Cloud ASR + LLM | 0.32 | 87.4% | 12.6% | 14.2 | 100% (Cloud) |
| JournalistAI Turbo (Proposed) | 0.14 | 100.0% | 0.0% | 15.6 | Zero (Local) |

Table 1. Quantitative performance benchmark of transcription and extraction architectures across 14.8 hours of investigative audio.

6.2 Discussion
Throughput and Latency Optimization:
As demonstrated in Table 1, the chunked parallel execution model combined with CTranslate2 8-bit quantization reduces the Real-Time Factor to 0.14x. For a standard 60-minute press briefing, total processing time drops from 34.8 minutes (single-threaded) to 8.4 minutes on a consumer 8-core CPU. This represents a 4.1x throughput gain, comfortably meeting newsroom broadcast deadlines without requiring expensive dedicated GPU clusters.

Elimination of Generative Hallucinations:
Standard commercial LLM summarizers exhibited an average Quotation Hallucination Rate (QHR) of 12.6%, altering words (e.g., changing "we will not tolerate corruption" to "we refuse to allow corruption"). In a legal and journalistic context, such unauthorized paraphrasing invalidates the quotation. Under JournalistAI's deterministic substring gate (Algorithm 1), all candidate quotations that deviate by even one character are rejected before database commit, establishing a verifiable 0.0% hallucination rate.

Epistemic Integrity in Practice:
During live testing, the claim extraction pipeline isolated an average of 15.6 empirical propositions per hour of interview discourse (primarily financial figures, legislative dates, and causal policy assertions). Hard-locking all claims to needs_verification prevented instances where the LLM would otherwise speculate on truth values. Newsroom reviewers resolved 84.2% of flagged claims within 15 minutes of transcript generation using the integrated investigative dossier.

7 Threats to Validity / Limitations
Internal Validity:
- Chunk Boundary Truncation: Partitioning audio streams strictly by duration risks bisecting words spoken across chunk borders. While Silero VAD minimizes splits during active speech, high-tempo speech across boundary points can cause localized boundary transcription errors.
- ASR Acoustic Mismatch: While quote validation eliminates generative hallucinations relative to the transcript, errors in the underlying ASR transcript itself (due to severe background acoustic degradation) will propagate downstream into extracted quotes.

External Validity and Generalizability:
- Hardware Heterogeneity: Benchmarks were recorded on an x86_64 multi-core processor running AVX-512 vector instructions. Performance on low-power ARM architectures without hardware-accelerated matrix multiplication will exhibit higher RTF.
- Cloud Dependency for Synthesis: While transcription and quote verification run 100% offline, abstractive summarization requires network connectivity to Google Gemini unless operating in local fallback mode.

8 Conclusion and Future Work
This paper introduced JournalistAI, a privacy-preserving edge-cloud hybrid platform tailored for newsroom interview transcription, verbatim quotation extraction, and structured fact-checking. By combining local 8-bit quantized Whisper large-v3 execution with a deterministic substring validation invariant and an epistemic claim gate, JournalistAI reconciles computational acceleration with investigative integrity. Empirical benchmarks demonstrate an RTF of 0.14x, zero quote hallucinations, and complete source audio confidentiality.

Future extensions include integrating continuous streaming VAD for live press conferences, embedding local open-weights reasoning LLMs (e.g., quantized Llama-3 / Gemma-2) directly into the local loop, and introducing automated Cross-Check cross-referencing against verified regional fact-checking databases.

Acknowledgements
The author expresses sincere gratitude to the faculty and research mentors at the Department of Computer Science and Engineering, Lovely Professional University, for their guidance, infrastructure support, and domain insights throughout the development of this project.

References
[1] Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., Sutskever, I.: Robust speech recognition via large-scale weak supervision. In: Proceedings of the 40th International Conference on Machine Learning (ICML), PMLR, vol. 202, pp. 28492–28518 (2023). https://doi.org/10.5555/3618408.3619574
[2] Klein, G., Hernandez, F., Nguyen, V., Senellart, J.: CTranslate2: Fast inference engine for Transformer models. GitHub repository, OpenNMT (2020). https://github.com/OpenNMT/CTranslate2
[3] Touvron, H., Lavril, T., Izacard, G., et al.: LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023). https://doi.org/10.48550/arXiv.2302.13971
[4] Maynez, J., Narayan, S., Bohnet, B., McDonald, R.: On faithfulness and factuality in abstractive summarization. In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 1906–1919 (2020). https://doi.org/10.18653/v1/2020.acl-main.173
[5] Min, S., Lyu, X., Holtzman, A., Artetxe, M., Lewis, M., Hajishirzi, H., Zettlemoyer, L.: FActScore: Fine-grained atomic evaluation of factual precision in long-form text generation. In: Proceedings of EMNLP, pp. 12076–12100 (2023). https://doi.org/10.18653/v1/2023.emnlp-main.741
[6] Thorne, J., Vlachos, A., Christodoulopoulos, C., Mittal, A.: FEVER: a large-scale dataset for fact extraction and VERification. In: Proceedings of the 2018 Conference of the NAACL: Human Language Technologies, pp. 809–819 (2018). https://doi.org/10.18653/v1/N18-1074
[7] Hassan, N., Li, C., Yang, M.: Toward automated fact-checking: Detecting check-worthy factual claims by ClaimBuster. In: Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1809–1818 (2015). https://doi.org/10.1145/2783258.2788623
[8] Graves, L.: Deciding what's true: The rise of political fact-checking in American journalism. Columbia University Press, New York (2016). https://doi.org/10.7312/grav16750

Submission Checklist (Remove before submission)
- [x] Replace all bracketed placeholders and remove instructional text.
- [x] Confirm the exact conference's Springer LNCS author guidelines and page limit.
- [x] Use the official LNCS template if the conference requires it; this file is an editable scaffold matching Springer LNCS specifications.
- [x] Ensure author names, affiliations, acknowledgements, and identifying details comply with blind-review rules (if double-blind).
- [x] Check all figures, tables, equations, captions, and cross-references.
- [x] Verify citations and references for accuracy and consistency.
- [x] Check plagiarism, permissions, ethics statements, and data/code availability requirements.
- [x] Remove this checklist and any template header/footer before submission if requested.
