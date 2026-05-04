/badge/JavaScript-ES6+-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)

## 🔄 App Workflow

```
Button Press → State Update (useState) → Expression Evaluation → Display Result
```

### 1️⃣ Display Panel
- Shows current input and running expression in real time
- Handles multi-digit number entry and decimal points
- Updates reactively via `useState` on every button press

### 2️⃣ Arithmetic Operations
- **Addition** `+` — sums two operands
- **Subtraction** `-` — finds the difference
- **Multiplication** `×` — computes the product
- **Division** `÷` — performs safe division

### 3️⃣ Utility Controls
- `AC` — clears all input and resets calculator state
- `DEL` — deletes the last entered character
- `=` — evaluates the full expression and renders the result
- `.` — appends a decimal point for float input

---

## 🔍 Key Highlights

- ⚛️ **React `useState` hook** — all calculator state (input, expression, result) managed cleanly with hooks
- 🎨 **CSS Grid layout** — button grid built with CSS Grid for a pixel-perfect, aligned calculator face
- 📱 **Responsive design** — adapts cleanly across desktop and mobile screen sizes
- 🧩 **Component-based** — calculator broken into reusable React components (Display, Button, ButtonGrid)
- ⚡ **Zero dependencies** — no external libraries beyond React itself

---

## 🗂️ Repository Structure

```
Calculator/
│
├── src/
│   ├── components/
│   │   ├── Calculator.jsx     # Main calculator component
│   │   ├── Display.jsx        # Expression and result display
│   │   └── Button.jsx         # Individual button component
│   ├── App.jsx                
│   ├── App.css                
│   └── main.jsx               
│
└── README.md
```

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/ronakrajput8882/Calculator.git
cd Calculator

# Install dependencies
npm install

# Start the dev server
npm run dev
```

> Visit `http://localhost:5173` in your browser.

---

## 🧠 Key Learnings

- Managing UI state with React's `useState` hook — tracking input, operator, and result as separate pieces of state
- Building a responsive grid layout with CSS Grid for the button matrix
- Handling edge cases in calculator logic — chaining operators, decimal point validation, division by zero
- Structuring a small React app cleanly with component separation even for simple UIs

---

## 🛠️ Tech Stack

| Tool | Purpose |
|:---|:---|
| React.js | UI framework and component logic |
| JavaScript (ES6+) | Arithmetic logic and event handling |
| CSS3 | Styling, grid layout, responsiveness |
| Vite | Development server and build tool |

---

<div align="center">

### Connect with me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ronakrajput8882)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwithronak)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ronakrajput8882)



<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=100&section=footer" width="100%"/>

</div>
