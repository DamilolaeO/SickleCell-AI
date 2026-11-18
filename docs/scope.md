# Scope (v1)

| The app DOES | The app DOES NOT |
|---|---|
| Explain what sickle cell disease is and how it is inherited | Diagnose sickle cell or any other condition |
| Explain what a pain crisis is and common triggers | Tell a user whether their current symptoms are a crisis |
| Explain complications (e.g. acute chest syndrome, stroke) in general terms | Assess how serious a user's symptoms are (triage) |
| Explain what common treatments are (e.g. hydroxycarbamide, transfusions, gene therapy) | Recommend, start, stop or change any treatment |
| Explain what common blood tests measure | Interpret a user's own test results |
| Explain general self-care (hydration, warmth, rest) using NHS wording | Give medication doses or painkiller schedules |
| Help users prepare general questions for appointments | Interpret journal entries or say if pain is "getting worse" |
| Let users record, edit, delete and export pain entries | Send journal data to the AI model |
| Show factual summaries of what the user recorded | Give country-specific service advice outside the UK in v1 (non-UK users get general information and a "contact your local emergency services" message) |
| Show a fixed urgent-care message when red-flag symptoms are mentioned | Generate its own emergency advice with the AI model |
| Cite the source for every educational answer | Answer when no trusted source covers the question (it says "I don't know") |

## Who it's for
- **Adults** living with sickle cell
- **Parents and carers** managing a child's sickle cell
- **Young people (13 to 17)** using it themselves
- **Children under 13** through a parent or carer's account, not their own
- **Users anywhere in the world.** The education content applies everywhere. Emergency messages and service advice are UK-specific in v1, so non-UK users are told to contact their local emergency services.

## Safeguards for these users
- **Child red flags.** Children have extra emergency signs (for example fever, a swollen tummy, sudden paleness or floppiness). The safety layer covers these as well as adult ones.
- **Children's data.** Under UK GDPR, under-13s need a parent's consent, and the app follows the ICO Children's Code (high privacy by default, plain language, no nudging).
- **Child-friendly wording.** Answers are written so young people can understand them.
- **Country selection at sign-up.** The emergency message matches where the user is.
