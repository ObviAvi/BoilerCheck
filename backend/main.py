from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/check")
def check():
    return {
        "answer": "Certain small appliances are allowed in Purdue University residences, such as microwaves, mini refrigerators, and coffee makers. However, high-heat or open-coil appliances like hot plates, space heaters, and toaster ovens are typically not allowed for safety reasons.",
        "documents": [
            {
                "document_id": "purdue_housing_safety_1",
                "title": "University Residences – Safety Guidelines",
                "domain": "housing",
                "url": "https://www.housing.purdue.edu/my-housing/info/general/residence-hall-guidelines.html",
                "effective_date": "2024-08-15",
                "sections": [
                    {
                        "section_title": "Allowed Appliances",
                        "text": "Residents may use approved small appliances such as microwaves and mini refrigerators that meet university safety standards."
                    },
                    {
                        "section_title": "Prohibited Appliances",
                        "text": "Appliances with open heating elements such as hot plates, space heaters, and toaster ovens are not permitted in residence halls."
                    }
                ]
            },
            {
                "document_id": "purdue_fire_safety",
                "title": "Purdue Fire Safety – Residence Hall Policies",
                "domain": "safety",
                "url": "https://www.purdue.edu/ehps/fire-safety/",
                "effective_date": "2024-08-15",
                "sections": [
                    {
                        "section_title": "Electrical Safety",
                        "text": "Only approved electrical appliances are permitted in residence halls to reduce fire risk."
                    },
                    {
                        "section_title": "Restricted Equipment",
                        "text": "High-wattage and heat-producing devices that pose safety hazards are restricted in student housing."
                    }
                ]
            }
        ]
    }