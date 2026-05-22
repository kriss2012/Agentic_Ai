**the extended output:**

**index.html**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">        
    <title>Vision Computers</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <nav>
            <ul>
                <li><a href="#home">Home</a></li>
                <li><a href="#services">Services</a></li>
                <li><a href="#repair">Repair</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </nav>
    </header>
    <main>
        <section id="home">
            <h1>Welcome to Vision Computers</h1>
            <p>We provide top-notch computer services, from repair to maintenance.</p>
            <img src="images/computer.jpg" alt="Computer Image">
        </section>
        <section id="services">
            <h1>Our Services</h1>
            <ul>
                <li><a href="#repair">Computer Repair</a></li>
                <li><a href="#upgrade">Computer Upgrade</a></li>
                <li><a href="#maintenance">Computer Maintenance</a></li>
                <li><a href="#consulting">Computer Consulting</a></li>
            </ul>
        </section>
        <section id="repair">
            <h1>Computer Repair</h1>
            <p>We repair all types of computers, from laptops to desktops.</p>    
            <img src="images/repair.jpg" alt="Repair Image">
            <button>Get a Quote</button>
        </section>
        <section id="upgrade">
            <h1>Computer Upgrade</h1>
            <p>We upgrade your computer to the latest hardware and software.</p>  
            <img src="images/upgrade.jpg" alt="Upgrade Image">
            <button>Get a Quote</button>
        </section>
        <section id="maintenance">
            <h1>Computer Maintenance</h1>
            <p>We provide regular maintenance to keep your computer running smoothly.</p>
            <img src="images/maintenance.jpg" alt="Maintenance Image">
            <button>Get a Quote</button>
        </section>
        <section id="consulting">
            <h1>Computer Consulting</h1>
            <p>We provide expert consulting services to help you make informed decisions.</p>
            <img src="images/consulting.jpg" alt="Consulting Image">
            <button>Get a Quote</button>
        </section>
        <section id="contact">
            <h1>Get in Touch</h1>
            <p>Contact us for any questions or concerns.</p>
            <form>
                <input type="text" placeholder="Name">
                <input type="email" placeholder="Email">
                <textarea placeholder="Message"></textarea>
                <button>Send</button>
            </form>
        </section>
    </main>
    <footer>
        <p>&copy; 2023 Vision Computers</p>
    </footer>
    <script src="script.js"></script>
</body>
</html>
```

**styles.css**
```css
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
}

header {
    background-color: #333;
    color: #fff;
    padding: 1em;
    text-align: center;
}

nav ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    justify-content: space-between;
}

nav li {
    margin-right: 20px;
}

nav a {
    color: #fff;
    text-decoration: none;
}

main {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2em;
}

section {
    background-color: #f7f7f7;
    padding: 2em;
    margin-bottom: 20px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

h1 {
    font-size: 24px;
    margin-bottom: 10px;
}

p {
    margin-bottom: 20px;
}

img {
    width: 100%;
    height: 200px;
    object-fit: cover;
    margin-bottom: 20px;
}

button {
    background-color: #333;
    color: #fff;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    background-color: #555;
}

footer {
    background-color: #333;
    color: #fff;
    padding: 1em;
    text-align: center;
    clear: both;
}
```

**script.js**
```javascript
// Get all buttons
const buttons = document.querySelectorAll('button');

// Add event listener to each button
buttons.forEach(button => {
    button.addEventListener('click', () => {
        // Get the section that the button is in
        const section = button.parentNode;

        // Get the id of the section
        const sectionId = section.id;

        // Scroll to the section
        document.getElementById(sectionId).scrollIntoView({
            behavior: 'smooth',
            block: 'center',
            inline: 'center'
        });
    });
});
```

**images**

* computer.jpg: a high-quality image of a computer
* repair.jpg: a high-quality image of a computer repair
* upgrade.jpg: a high-quality image of a computer upgrade
* maintenance.jpg: a high-quality image of a computer maintenance
* consulting.jpg: a high-quality image of a computer consulting

Note: You will need to replace the image files with your own images.

This code creates a basic website for Vision Computers with the following pages:  

* Home: an introduction to the company
* Services: a list of services offered by the company
* Repair: information about computer repair services
* Upgrade: information about computer upgrade services
* Maintenance: information about computer maintenance services
* Consulting: information about computer consulting services
* Contact: a form to contact the company

The website uses HTML, CSS, and JavaScript to create a responsive design that adapts to different screen sizes. The JavaScript code adds an event listener to each button, so that when a button is clicked, the corresponding section is scrolled into view.




pdf data : Title: Scholarship Information 2025
1. Eligibility:
- Open to students in India pursuing undergraduate degrees.
- Annual family income must be below ₹6,00,000.
- Minimum 60% marks in the last qualifying exam.
2. Documents Required:
- Income certificate
- Aadhaar card
- Bank passbook
- Marksheet
3. Deadline: October 15, 2025
4. Benefits:
- ₹10,000 per semester for tuition
- Book allowance of ₹3,000 per year
5. How to Apply:
Visit https://scholarships.gov.in and register under the NSP portal.


RAG for developer portfolio

Building a RAG-powered chatbot for your developer portfolio is a massive flex. It immediately shows recruiters and visitors that you don't just write code—you understand how to build and deploy modern AI architectures.

Since this bot will be interacting with potential employers, clients, or other developers, it needs to be highly accurate, professional, and entirely focused on you and your work. It cannot hallucinate or get distracted by random internet questions.

Here is the blueprint for adapting the RAG concept into a professional portfolio chatbot.

1. Curate Your "Brain" (The Knowledge Base)
Instead of feeding the system a generic PDF like the scholarship example, you need to build a custom knowledge base containing all your professional data. Create text or Markdown files for the following:

The Master Resume: Education (like your BCA coursework and how it emphasizes professional software engineering practices), core skills (Python, React, Flask, C++), and contact info.

Deep-Dive Project Readmes: This is where RAG shines. Create detailed documents for your major builds.

For AI/ML work, detail the NLP models and backend logic used in systems like BhashaConvert or your Fake Review Detection system.

For full-stack work, document the component structure and Tailwind CSS styling choices for things like your Telegram clone or LearnSphere.

For game dev, break down the 3D logic and engine concepts you've explored.

The "Why" and "How": Recruiters don't just care about the code; they care about the problem-solving. Write down the challenges you faced building platforms like EchoRepears or the AI Placement Coach and how you overcame them.

2. Processing and Vector Storage
The technical pipeline remains almost exactly the same as your previous script:

Chunking: Use RecursiveCharacterTextSplitter to break your project docs and resume down into small, digestible paragraphs.

Embeddings: Convert those chunks into vectors (using Google Generative AI embeddings or similar).

Vector Database: Store them in a persistent Chroma DB.

3. The "Strict Professional" System Prompt
This is the most critical part of a public-facing portfolio bot. You must use a system prompt that gives the LLM a distinct persona and strict boundaries so visitors can't trick it into acting like a generic ChatGPT clone.

When setting up your LangChain prompt template, use something like this:

"You are the official AI portfolio assistant for Krishna. Your job is to answer questions from recruiters and visitors about Krishna's coding projects, software engineering skills, and education. You are professional, concise, and enthusiastic.

Use ONLY the provided context to answer the question. If a visitor asks a coding question unrelated to Krishna's work, politely decline and steer the conversation back to Krishna's portfolio. If you do not know the answer based on the context, say: 'I don't have the specifics on that, but you can reach out to Krishna directly at [Email/LinkedIn]!'"

4. The Frontend Experience
Since you have solid experience with React and Tailwind CSS, you can build a sleek chat interface that sits in the corner of your portfolio site.

Fast Responses: Use a fast, low-latency model (like Llama-3-8b via Groq, which you are already using) so the chat feels snappy to visitors.

Guided Prompts: Don't just leave an empty text box. Give visitors clickable suggested questions to get them started, such as:

"What is your tech stack?"

"Tell me about your AI and NLP projects."

"What are you currently studying?"

By feeding the RAG pipeline your specific project architectures and academic background, you guarantee that when a recruiter asks, "What is your experience with React?", the bot won't give a generic definition of React. It will actually pull up the exact details of the web apps you've built.









answer of the agent responce given above:

'weather_descriptions': ['Sunny'], 'astro': {'sunrise': '05:49 AM', 'sunset': '07:02 PM', 'moonrise': '11:17 AM', 'moonset': 'No moonset', 'moon_phase': 'Waxing Crescent', 'moon_illumination': ', tool_call_id='4wcf285dq', AIMessage(content='The current weather in Pachora is sunny with a temperature of 37 degrees Celsius.'),}