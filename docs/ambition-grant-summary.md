Strata — Affordable, Honest Precision Agriculture for Small Farms

Ambition Grant application, Kurt Giessler Foundation for Youth Achievement

Applicants: [Names], [grades/ages], [school], Chester County, PA
Project repository and full technical writeup available on request.


In one paragraph

Big precision-agriculture tools can tell a farmer what their field will yield and what to add to the soil, but they are priced for operations running thousands of acres, and the small farms around Exton and Malvern can't really touch them. Strata is a working software system we built that answers those same questions for a small farm using only free public data and no expensive hardware. It already runs end to end: it predicts crop yield from soil and weather, it gives fertilizer recommendations backed by cited Penn State agronomy rules, and, unusually, it is built to say "I don't know, get a soil test" when it honestly can't answer. Along the way we found and proved a specific limit on how good the predictions can get with public data alone, and that proof points directly at the one thing that would break through it: a cheap soil sensor taking real readings in a real field over time. The next step is a simple, affordable soil device a farmer can carry for a quick reading or leave planted to log a stretch of the season, sending its readings to an app so they can see everything on one dashboard. It turns a rigorous software result into a tool a farmer can actually use.


The problem we set out to solve

Small farms are getting left behind by the technology that is transforming big agriculture. A grower with a few hundred acres in Chester County faces the same questions a 5,000-acre operation does. What will my field yield this year? What should I add to my soil, and how much? Can I trust a cheap sensor instead of paying for expensive lab tests?

The platforms that answer these are subscription products aimed at industrial farms. Our goal was to answer the same questions with tools a small farm could actually afford, ideally for free, and to be genuinely honest about which answers the data can and can't support. That last part matters more than it sounds. Getting a fertilizer rate wrong costs a farmer real money and pollutes real water, so a tool that confidently guesses is worse than no tool at all.


What we built, and it works today

Strata is a complete, runnable system, not a proposal or a prototype sketch. It pulls from four free public datasets, USDA crop yields, USDA soil surveys, USDA lab soil profiles, and NASA/ORNL daily weather, and combines them into one analysis table covering 66 Pennsylvania counties over 24 years. On top of that it does four things, all of which run today.

It predicts yield: a machine-learning model predicts a county's crop yield for a given year from its soil and that season's weather. It identifies soil as a rigor test: a separate model shows that cheap, measurable soil numbers really do carry meaningful soil information, which is the assumption the whole product rests on. It recommends fertilizer: a transparent engine applies existing, cited Penn State agronomy rules to a field's soil test and prints advice, with the published source rule attached to every recommendation. And it has a calibration seam, the one clearly-defined place where a cheap soil sensor plugs in, which is the hardware that comes next.

Every result is checked on real, held-out data. Every data source is documented and reproducible. Nothing is faked; where we use placeholder data to test an interface, it is labeled as such and never reported as a real finding. We were strict about this on purpose, because it is the difference between a science project and a demo.


The results, reported honestly

We want to be straight about the numbers, because the honesty is the project. On the corn yield model, for counties the model has never seen, it predicts yield with an average error of about 14.5 bushels per acre, versus 22.8 for just guessing the county average, a real and meaningful improvement. On soil identification, the model recovers a soil's scientific classification from cheap lab numbers with about 70 percent accuracy, versus 34 percent for guessing the most common type, roughly double. And the feature it leans on most is clay increase with depth, which is exactly the signal real soil scientists use, so the model rediscovered genuine soil science on its own. On fertilizer, the engine refuses to guess: when a farmer hasn't given it the specific lab measurements it needs, it returns "obtain a soil test" instead of inventing a number, and that refusal is a designed feature, not a gap.

These aren't headline-grabbing numbers, and we're not pretending they're state-of-the-art. What makes the project strong is that we can explain exactly why they land where they do, which brings us to the discovery at the heart of this application.


The discovery: a proven ceiling, and the exact way through it

Instead of accepting a mediocre score and shrugging, we measured why the model can't do better. We split all the variation in corn yield into two parts. About 20 percent comes from which county you are in, since some places have better soil. About 80 percent comes from which year it is, meaning weather and improving farm practices.

Here is the key part. The soil data we have is a single fixed snapshot per county. Chester County's soil pH is one number, the same in 2003 as in 2023. A number that never changes can't possibly explain something that changes every year. So all our soil features are competing for at most a fifth of the problem, and they are structurally incapable of touching the other four-fifths. We tested six different ways to improve the model, and none of them helped, exactly as this analysis predicts.

That turns "our model is mediocre" into something much more useful: a precise statement of what kind of new information would actually help. The answer is a soil measurement that changes over time and place, real readings taken in a real field, logged through the season and taken again the next year. That is a cheap soil sensor left to record in the field. Our software didn't just leave room for hardware; it derived the exact specification for the hardware that would break its ceiling. The next real gain isn't a smarter algorithm, it is a better measurement, and we know precisely what that measurement has to be.


Why this project is ambitious

It is honest where most projects overstate. We built refusal, uncertainty, and a named quality threshold right into the system. It withholds a forecast when it wouldn't beat guessing the average, and it tells the farmer so. For a tool that touches someone's livelihood and the local watershed, knowing when not to answer is the hard part, and the responsible one.

It doesn't reinvent solved science. Fertilizer rules come from decades of Penn State field trials. We encode and cite them instead of trying to out-guess them with our own data, so an extension agronomist could audit every recommendation line by line.

It is reproducible and rigorous. Fixed random seeds, documented data provenance, every dropped row counted and explained, and even a bug we found and documented with its correction rather than quietly erasing. That is how real science is supposed to work.

And the roadmap is derived, not assumed. The hardware isn't a "wouldn't it be cool" add-on. It is the mathematically-identified next step.


What comes next: the field device and the app

The next step carries Strata from a proven software result into a tool a farmer can actually use in the field. It is an affordable soil device a farmer can either carry from spot to spot for a quick reading or plant in the ground and leave in place, so it records real readings over time rather than a single snapshot. Those readings change over the season and from spot to spot, which is exactly the missing ingredient our ceiling analysis identified. Each cheap sensor is corrected onto a trusted reference scale by our existing calibration module, using standard calibration solutions and the soil reports the farms already have rather than a fresh lab test for every reading, so an inexpensive sensor becomes something you can actually trust.

The device also records where each reading was taken, so every measurement is tagged with its place in the field. That means a farmer can find the device again after it has been sitting in tall corn for weeks, and it lets Strata line up readings from different spots and different visits into a real picture of the field rather than one undated number. The device sends its calibrated, located readings to a phone, and from there into Strata's dashboard, so the farmer sees their yield estimate, their fertilizer advice, and how their soil is doing all in one place, with no spreadsheets and no lab turnaround. The dashboard already exists in the software; the device is what finally feeds it live field data.

The sensors are chosen deliberately, each one addressing a specific gap in the current system. Soil moisture and temperature capture water and heat in the root zone, and because the device can log them over time, they are exactly the time-varying, within-field signal our analysis proved is the only thing that can address the roughly 80 percent of yield variation the static public data can't touch. Soil pH sets how available nutrients are to a crop, so it lets the recommender assess lime need in the field instead of relying only on a coarse county soil survey. Nitrogen, phosphorus, and potassium readings directly target the measurements the recommender currently refuses to guess; cheap nutrient sensors are only reference-grade, which is exactly why they run through the calibration seam and are checked against soil reports the farms already have, the honesty discipline of the project applied to hardware. And a soil carbon dioxide sensor adds something most commercial probes leave out entirely: the CO2 coming off the soil reflects the microbial life breathing inside it, a direct early reading of biological soil health that pure chemistry misses.

Every one of these plugs into the same software with nothing upstream changing, because the calibration seam was built for this handoff. And importantly, the software has zero dependency on this hardware to be useful today. The device is an accuracy layer, not a prerequisite. The core project already works; the hardware is there to push past a limit we have already measured, not to get the project off the ground.


Impact, and why it's worth funding

If it works, Strata gives a small Pennsylvania farmer, at almost no cost, a credible estimate of what their field will yield and exactly how much fertilizer to apply, no more and no less. Applying less unnecessary fertilizer isn't only cheaper for the farmer; it keeps phosphorus out of local streams and the Chesapeake watershed, which is a direct environmental win for our region. And the whole thing is auditable and free, so it can be shared rather than sold.

More personally, we set out to answer a real question for the farms in our own county, and we held ourselves to the standard of saying plainly what we don't know. That led us to a genuine discovery, a measured ceiling and the exact way through it. This grant is the difference between a rigorous result on paper and a device a farmer can use in a Chester County field. We have done the hard analytical work to know exactly what to build and why it matters. We are asking for the means to go build it.


Full technical documentation, a plain-language walkthrough (HOW-IT-WORKS.md) and a formal methods writeup (METHODS.md), is available and kept in sync with the working code.
