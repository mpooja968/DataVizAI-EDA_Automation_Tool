from reportlab.lib.pagesizes import letter # type: ignore
from reportlab.pdfgen import canvas # type: ignore
import io

def generate_pdf(content, file_name="EDA_Report.pdf"):
    buffer = io.BytesIO()  # Create a buffer to store the PDF
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter  # Get page size

    pdf.setFont("Helvetica", 12)  # Set font
    y_position = height - 40  # Initial Y position for text

    # Write content line by line
    for line in content.split("\n"):
        pdf.drawString(50, y_position, line)
        y_position -= 20  # Move down for the next line

        # If Y position is too low, create a new page
        if y_position < 40:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y_position = height - 40

    pdf.save()  # Save the PDF to the buffer
    buffer.seek(0)  # Move to the start of the buffer
    return buffer
