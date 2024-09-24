import faiss
import requests
import numpy as np

dim = 4096

titles = [
'The Rise of Quantum Computing: What’s Next?', '5G vs. 6G: The Future of Connectivity', 'How Blockchain is Reshaping Data Security', 'The Ethics of Artificial Intelligence in Everyday Life', 'The Role of Cloud Computing in Modern Business', 'Emerging Trends in Cybersecurity for 2025', 'The Impact of Virtual Reality in Education', 'Decentralized Web: A New Internet for All?', 'Autonomous Vehicles: Technology and Challenges', 'The Growth of Edge Computing in Smart Cities', 'AI in Healthcare: From Diagnosis to Treatment', 'The Future of Human-AI Collaboration', 'Reinforcement Learning: Breakthroughs and Applications', 'AI and Ethics: Navigating Moral Boundaries', 'AI in Agriculture: Growing a Smarter Future', 'Language Models: How GPT is Changing Content Creation', 'AI in Retail: Personalizing the Customer Experience', 'The Role of AI in Predicting Natural Disasters', 'AI in Financial Markets: The Rise of Robo-Advisors', 'Bias in Machine Learning Models: A Deep Dive', 'How to Scale a Startup in a Competitive Market', 'The Top Startup Hubs Around the World', 'Navigating the VC Funding Landscape in 2024', 'Startups in AI: Innovating the Next Generation of Tools', 'How to Build a Resilient Startup Culture', 'Sustainability Startups: Solving the Climate Crisis', 'From Idea to IPO: The Journey of Successful Founders', 'Key Legal Pitfalls to Avoid in Startup Growth', 'Startup Failure Rates: Why 90% Don\'t Survive', 'Creating Disruption: How Startups Redefine Industries', 'Digital Transformation: Why Every Business Needs It', 'The Importance of Corporate Social Responsibility in 2024', 'Navigating Global Trade Disruptions for Business Resilience', 'The Role of Remote Work in Business Success', 'How AI is Driving Efficiency in Business Processes', 'Business Model Innovation in the Age of AI', 'The Future of Work: Trends to Watch in Business Strategy', 'The Role of Data Analytics in Modern Business Decision-Making', 'Building Brand Loyalty in the Digital Age', 'The Impact of Automation on Job Markets and Businesses', 'Inflation in 2024: Global Trends and Impacts', 'The Economics of Climate Change: Costs and Opportunities', 'Digital Currencies: The Future of Money?', 'Global Supply Chain Disruptions and Economic Recovery', 'The Impact of AI on Labor Markets and Economic Policy', 'How Cryptocurrency is Shaping Global Economic Policy', 'Global Recession: Lessons from the Past and Future Outlook', 'Economic Growth in Developing Countries: Trends and Challenges', 'Trade Wars: The Impact of US-China Relations on Global Economics', 'The Role of Central Banks in Stabilizing Global Markets', 'Telemedicine: Revolutionizing Healthcare in Rural Areas', 'The Role of AI in Early Disease Detection', 'Genomics and Personalized Medicine: A New Era in Healthcare', 'The Global Mental Health Crisis: Solutions and Innovations', 'The Future of Wearable Health Tech: From Fitness to Disease Prevention', 'How Technology is Improving Healthcare Accessibility', 'AI in Drug Discovery: Accelerating Medical Research', 'Data Privacy in Healthcare: Challenges and Solutions', 'The Rise of Virtual Health Assistants: A New Standard of Care', 'Healthcare Systems Post-Pandemic: Lessons Learned and Innovations', 'Cyber Warfare: The New Battleground', 'The Evolution of Military Drones: From Surveillance to Combat', 'AI and Autonomous Weapons: The Future of Warfare?', 'Space Warfare: Emerging Threats and Global Policies', 'The Role of Technology in Modern Intelligence Gathering', 'Hybrid Warfare: Blending Conventional and Cyber Tactics', 'The Impact of Economic Sanctions on Global Conflicts', 'How Warfare Shapes Global Diplomacy in the 21st Century', 'Robotics in Modern Combat: Strengths and Ethical Concerns', 'War and Peace: The Changing Role of International Organizations', 'The Future of Smart Cities: Innovations in Urban Infrastructure', 'Renewable Energy Infrastructure: The Key to Sustainable Growth', 'How 5G Will Transform Global Infrastructure', 'Challenges in Rebuilding After Natural Disasters', 'The Role of AI in Traffic Management and Urban Planning', 'Sustainable Construction: New Materials and Methods', 'The Evolution of Global Transportation Systems', 'Smart Grids: The Next Frontier in Energy Distribution', 'The Impact of Climate Change on Global Infrastructure', 'The Role of IoT in Modernizing Public Infrastructure', 'AI in Sports: How Technology is Changing the Game', 'The Business of Sports: Global Revenue Streams in 2024', 'Wearable Tech in Sports: Monitoring Performance and Health', 'The Role of Data Analytics in Competitive Sports Strategy', 'Women in Sports: Breaking Barriers and Driving Change', 'Esports: The Rise of a Billion-Dollar Industry', 'The Evolution of Sports Sponsorships in the Digital Age', 'Mental Health in Sports: Addressing the Pressure to Perform', 'The Impact of Social Media on Athlete Branding', 'From Athletes to Entrepreneurs: The Rise of Sports Startups', 'Sustainable Fashion: The Movement Toward Eco-Friendly Clothing', 'The Role of AI in Fashion Design and Retail', 'The Rise of Digital Fashion: NFTs and Virtual Runways', 'How Technology is Shaping the Future of Fashion Retail', 'Fashion Influencers: Shaping Trends in the Digital Age', 'The Impact of Globalization on Fashion Trends', 'Streetwear: From Subculture to Mainstream Fashion', 'The Role of Fashion Weeks in a Post-Pandemic World', 'Gender-Neutral Fashion: Breaking Traditional Boundaries', 'Fashion Startups: Innovating in a Competitive Industry'

]

# Initialize FAISS index
index = faiss.IndexFlatL2(dim)

# Initialize empty array to store embeddings
X = np.zeros(shape=(len(titles), dim), dtype='float32')

# Iterate over titles to get embeddings
for i, title in enumerate(titles):
    res = requests.post(
        url='http://localhost:11434/api/embeddings',
        json={
            'model': 'llama3.1',
            'prompt': title
        }
    )
    try:
        embedding = res.json().get('embedding', None)
        if embedding:
            X[i] = np.array(embedding, dtype='float32')
        else:
            print(f"Embedding not found for title: {title}")
    except (KeyError, ValueError) as e:
        print(f"Error fetching embedding for {title}: {e}")

# Add the embeddings to the FAISS index
index.add(X)

# New prompt to search
new_prompt = 'Fighter gets UFC title'

# Get embedding for the new prompt
res = requests.post(
    url='http://localhost:11434/api/embeddings',
    json={
        'model': 'llama3.1',
        'prompt': new_prompt
    }
)

try:
    new_embedding = res.json().get('embedding', None)
    if new_embedding:
        embedding = np.array([new_embedding], dtype='float32')

        # Search for similar titles in the FAISS index
        D, I = index.search(embedding, k=3)

        # Print out the top 10 closest titles
        print(np.array(titles)[I.flatten()])
    else:
        print(f"Embedding not found for new prompt: {new_prompt}")
except (KeyError, ValueError) as e:
    print(f"Error fetching embedding for new prompt: {e}")
