\# DocIntel LayoutLMv3



An end-to-end document intelligence system for classifying document images using \*\*LayoutLMv3\*\*, \*\*Tesseract OCR\*\*, visual information, and document layout.



The system classifies documents into five categories:



\- Invoice

\- Resume

\- Form

\- Budget

\- Advertisement



The trained model is exposed through a \*\*FastAPI REST API\*\* and can be deployed as a \*\*CPU-only Docker container\*\*.



\---



\## Results



| Metric | Result |

|---|---:|

| Test Accuracy | \*\*93.42%\*\* |

| Test Loss | \*\*0.2150\*\* |

| Test Samples | \*\*365\*\* |

| Correct Predictions | \*\*341 / 365\*\* |

| Misclassified | \*\*24 / 365\*\* |



\### Per-Class Performance



| Class | Precision | Recall | F1 |

|---|---:|---:|---:|

| Invoice | 89.33% | 91.78% | 90.54% |

| Resume | 97.22% | 95.89% | 96.55% |

| Form | 90.28% | 89.04% | 89.66% |

| Budget | 92.11% | 95.89% | 93.96% |

| Advertisement | 98.57% | 94.52% | 96.50% |



\---



\## Architecture



```text

&#x20;                   Document Image

&#x20;                         |

&#x20;                         v

&#x20;                Image Preprocessing

&#x20;                    (Grayscale)

&#x20;                         |

&#x20;                         v

&#x20;                   Tesseract OCR

&#x20;                         |

&#x20;                +--------+--------+

&#x20;                |                 |

&#x20;                v                 v

&#x20;            OCR Words       Bounding Boxes

&#x20;                |                 |

&#x20;                +--------+--------+

&#x20;                         |

&#x20;                         v

&#x20;                LayoutLMv3 Processor

&#x20;                         |

&#x20;            +------------+------------+

&#x20;            |                         |

&#x20;            v                         v

&#x20;      Text + Layout              Image Features

&#x20;            |                         |

&#x20;            +------------+------------+

&#x20;                         |

&#x20;                         v

&#x20;                   LayoutLMv3

&#x20;                         |

&#x20;                         v

&#x20;               Document Classification

&#x20;                         |

&#x20;                         v

&#x20;       Invoice / Resume / Form / Budget /

&#x20;                 Advertisement



\---



\## How It Works



The system combines three sources of information for document classification:



\### 1. Text



Tesseract OCR extracts words from the document image.



\### 2. Layout



Each OCR word is associated with a bounding box representing its position on the document.



The bounding boxes are normalized to LayoutLMv3's `\[0, 1000]` coordinate system.



\### 3. Visual Information



LayoutLMv3 also processes the document image itself, allowing the model to use visual features together with text and spatial layout.



These features are processed together to classify the document.



\---



\## Technology Stack



| Component | Technology |

|---|---|

| Language | Python |

| Deep Learning | PyTorch |

| Transformer | LayoutLMv3 |

| Model Library | Hugging Face Transformers |

| OCR | Tesseract |

| OCR Wrapper | pytesseract |

| Image Processing | OpenCV |

| API | FastAPI |

| Server | Uvicorn |

| Testing | Pytest |

| Deployment | Docker |

| Version Control | Git |



\---



\## Document Classes



The classifier supports five document categories:



\- Invoice

\- Resume

\- Form

\- Budget

\- Advertisement



The test set contains \*\*365 documents\*\*, with \*\*73 samples per class\*\*.



\---



\## Evaluation



The model was evaluated on a held-out test set.



\### Overall Performance



| Metric | Result |

|---|---:|

| Test Accuracy | \*\*93.42%\*\* |

| Test Loss | \*\*0.2150\*\* |

| Correct Predictions | \*\*341 / 365\*\* |

| Misclassified | \*\*24 / 365\*\* |



\### Confusion Matrix



Rows represent actual classes and columns represent predicted classes.



```text

&#x20;                        invoice  resume  form  budget  advertisement



invoice                       67       0     2       3          1

resume                         1      70     1       1          0

form                           6       0    65       2          0

budget                         0       0     3      70          0

advertisement                  1       2     1       0         69



Error Analysis



The model misclassified 24 of 365 test documents.



The errors were analyzed using:



Actual class

Predicted class

Model confidence

OCR word count

Average OCR confidence



Across the misclassified documents:



Average OCR words:              70.38

Average OCR confidence:         46.31%

Documents with <10 OCR words:   4

Low OCR confidence documents:   15

Zero OCR documents:             0



This analysis showed that OCR quality contributes to several difficult classification cases.



Some incorrect predictions were also made with high model confidence, showing that OCR improvement alone will not completely eliminate classification errors.



OCR Preprocessing Experiment



Different OCR preprocessing strategies were compared on difficult test cases.



Strategy	Avg. OCR Words	Avg. OCR Confidence	Zero-word Documents

Adaptive Threshold	86.0	38.5%	3

Grayscale	86.6	50.3%	0

Original	86.6	50.3%	0



Grayscale preprocessing produced better OCR statistics than adaptive thresholding and was retained as the final preprocessing strategy.



FastAPI



The trained model is exposed through a FastAPI REST API.



Start the API

uvicorn app.main:app --reload



The API runs at:



http://127.0.0.1:8000

Health Check

Invoke-RestMethod http://127.0.0.1:8000/health



Example response:



{

&#x20; "status": "healthy",

&#x20; "service": "DocIntel LayoutLMv3"

}

Document Prediction



Endpoint:



POST /predict/



Example:



curl.exe -X POST "http://127.0.0.1:8000/predict/" `

&#x20; -F "file=@C:\\path\\to\\document.png"



Example response:



{

&#x20; "filename": "advertisement\_0040.png",

&#x20; "document\_type": "advertisement",

&#x20; "confidence": 0.9889517426490784,

&#x20; "ocr\_words": 0,

&#x20; "probabilities": {

&#x20;   "invoice": 0.005527077242732048,

&#x20;   "resume": 0.0019534314051270485,

&#x20;   "form": 0.0014448192669078708,

&#x20;   "budget": 0.0021229612175375223,

&#x20;   "advertisement": 0.9889517426490784

&#x20; }

}

API Validation



The API supports:



PNG

JPG

JPEG



Maximum upload size:



10 MB



The API also validates uploads, generates temporary filenames, handles inference errors, and removes temporary files after prediction.



Docker Deployment



The application can be deployed as a CPU-only Docker container.



The container includes:



Python

CPU PyTorch

Hugging Face Transformers

OpenCV

Tesseract

FastAPI

Uvicorn



The trained model checkpoint is mounted at runtime rather than copied into the Docker image.



Build

docker build -t docintel-layoutlmv3:cpu .

Run

docker run --rm -p 8000:8000 `

&#x20; -v "${PWD}\\checkpoints\\grayscale\\best\_model:/app/checkpoints/grayscale/best\_model:ro" `

&#x20; docintel-layoutlmv3:cpu

Test

Invoke-RestMethod http://127.0.0.1:8000/health



Prediction:



curl.exe -X POST "http://127.0.0.1:8000/predict/" `

&#x20; -F "file=@C:\\Dev\\docintel-layoutlmv3\\data\\clean\\test\\advertisement\\advertisement\_0040.png"



This demonstrates that the trained LayoutLMv3 inference service can be packaged and deployed as a portable CPU-based container.



Testing



Automated API tests are implemented using Pytest.



Run:



python -m pytest tests -v



Current tests cover:



Health endpoint

Prediction endpoint

Invalid file type rejection

Project Structure

docintel-layoutlmv3/

│

├── app/

│   ├── api/

│   ├── core/

│   ├── inference/

│   ├── layoutlm/

│   ├── ocr/

│   ├── schemas/

│   └── main.py

│

├── scripts/

│   ├── evaluate\_test.py

│   ├── analyze\_errors.py

│   ├── compare\_ocr\_preprocessing.py

│   ├── error\_analysis.py

│   ├── predict.py

│   ├── train.py

│   └── ...

│

├── tests/

│   └── test\_prediction\_api.py

│

├── data/

├── checkpoints/

├── Dockerfile

├── requirements.txt

├── requirements-docker.txt

├── .gitignore

└── README.md



Trained model checkpoints are excluded from Git because of their size.



Training



The project uses:



microsoft/layoutlmv3-base



Training configuration includes:



Batch size:                 2

Gradient accumulation:      4

Epochs:                     5

Learning rate:              2e-5

Weight decay:               0.01

Random seed:                42



The training configuration is centralized in:



app/layoutlm/training.py

Limitations

The classifier currently supports five document categories.

OCR quality can affect classification performance.

Some visually similar document types remain difficult to distinguish.

High-confidence incorrect predictions can still occur.

CPU inference is slower than GPU inference.

The current system performs document-level classification rather than key-value extraction.

The trained checkpoint is not stored in the Git repository.

Future Improvements

Add more document categories

Improve OCR preprocessing

Add document orientation correction

Add confidence calibration

Add key information extraction

Extract fields such as invoice number, date, total, and name

Add token-level classification

Add document search and retrieval

Add a frontend document upload interface

Add CI/CD

Add cloud deployment

Add model monitoring

Key Engineering Highlights



The project demonstrates an end-to-end ML engineering workflow:



Dataset Preparation

&#x20;       ↓

Image Preprocessing

&#x20;       ↓

OCR

&#x20;       ↓

Bounding Box Extraction

&#x20;       ↓

LayoutLMv3 Processing

&#x20;       ↓

Model Fine-Tuning

&#x20;       ↓

Test Evaluation

&#x20;       ↓

Error Analysis

&#x20;       ↓

FastAPI Inference

&#x20;       ↓

Automated Testing

&#x20;       ↓

Docker Deployment



Key concepts demonstrated:



Computer Vision

OCR

Document AI

Transformer models

Layout-aware modeling

PyTorch

Hugging Face Transformers

OpenCV

FastAPI

REST APIs

Docker

Automated testing

Model evaluation

Error analysis

Author



Abhinav Varma



GitHub:

https://github.com/abhinavv0516



Portfolio:

https://abhinavv0516.vercel.app







\### One thing I'd change from your current README





Don't add the huge per-class table \*\*twice\*\*. You already have it under `Results`, so keep that and don't repeat it under `Evaluation`.





Your current beginning is good:





\*\*Title → description → classes → results → architecture\*\*





That's exactly the right opening for a recruiter. The sections above





