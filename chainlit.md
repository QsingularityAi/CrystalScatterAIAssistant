# Welcome to CrystalScatter AI Assistant! 🚀🤖

Hi there! 👋 We're excited to have you on board. As your AI Assistant, I'll guide you through the essential aspects parameter config of your simulation and computation experiments."

# How I Can Help
As your AI assistant, I'm here to support you with:

- **cofigration Genrator:** Provide explanations and clarifications on the structure and functions within the source code file.
- **Customization Guidance:** Offer insights into how to modify parameters to meet specific requirements for your experiment.
- **Error Resolution:** Help troubleshoot errors in your script or setup.



# Example Prompts
To get started, feel free to ask me a question or provide some information about an issue you are facing. Here are few examples of how you can & Commands
This prompt Design using Few-shot-learning method 

Prompt Templete 👇

"Create a configuration file for a scientific experiment simulation with the following structure:

1. Start with a comment specifying the name of the experiment.
   Name of the experiment = "Experment name" 

2. For each parameter, provide:
   - A comment describing the parameter (if applicable)
   - An enable/disable flag (ena_[ParameterName])
   - The parameter value (val_[ParameterName])

3. Include parameters for:
   - Geometric properties (e.g., Alpha, Ay1, Ay2, Ay3, Az1, Az2, Az3)
   - Experimental setup (e.g., Base, Twinned, WAXS)
   - Particle properties (e.g., CBInterior, CBParticle, CBPeak)
   - Physical constants and measurements (e.g., Wavelength, Det, PixelNoX, PixelNoY)
   - Calculation parameters (e.g., HKLmax, I0)
   - Lattice properties (e.g., LType, uca, ucb, ucc, ucalpha, ucbeta, ucgamma)
   - Simulation control (e.g., GridPoints, BeamPos, generatePNG)

4. For most parameters, set the enable flag to false and provide a default value.

5. For critical simulation parameters like GridPoints, BeamPos, and generatePNG, set the enable flag to true.

6. Include parameters for calculation time and preparation time with enable flags set to true.

7. Specify the number of images to generate and the output path.

8. Use consistent formatting:
   - Comments start with #
   - Boolean values are represented as True/False
   - Numeric values are provided without units
   - String values are provided in curly braces with an index, e.g., disk {2}

9. Group related parameters together (e.g., all unit cell parameters, all pixel-related parameters).

Ensure that the configuration covers all aspects of the experiment, including particle properties, detector settings, lattice parameters, and simulation controls."


## Welcome screen

For additional support or complex inquiries, please contact the developer or consult the detailed documentation.

Welcome aboard, and happy engineering! 🚀💪



