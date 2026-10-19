# Carnegie Mellon University: AI & Vision Courses Reference

**Reviewed against official sources: October 18, 2026.** Use these courses to deepen a topic alongside the roadmap's runnable labs. Public notes do not imply enrollment, grading, unrestricted assignment access, or a complete free online course. The term named below identifies the material reviewed; moving course homepages can change next semester.

New to the subject? Start with [AI and robot learning foundations](0.%20AI%20and%20Robot%20Learning%20Foundations/Guide.md). You can train the supplied CPU policy before attempting a university semester's assignments.

## 1. 07-280: AI & Machine Learning I

The [current course page](https://www.cs.cmu.edu/~07280/) is **Fall 2026**. The 07-280/07-380 sequence replaces the retired 15-281/10-315 courses. It combines search, supervised learning, neural networks, language models, reinforcement learning and ethics; the syllabus includes AlexNet, GPT-2 and AlphaZero implementation assignments. Later Fall assignments were still scheduled work on the review date, not a complete released package.

The [FAQ and prerequisite comparison](https://www.cs.cmu.edu/~07280/#faq) require prior 15-122, linear algebra and mathematical foundations, with probability and Calculus 2 as corequisites. **10-301 is not an automatic substitute** for 07-280 in the AI core or as the prerequisite for 07-380; the FAQ describes an advising pathway for students who already took it.

**What you can use:** public lecture PDFs, notes and released worksheets. Canvas/Gradescope participation is separate. AIMA and some library-linked readings require access; “online” does not mean every book is freely downloadable. The [public notation guide](https://www.cs.cmu.edu/~07280/notes/Notation_Guide.pdf) is a useful first reading. For enrollment, use the current page's grading, collaboration and submission rules rather than an old copied timetable.

**Roadmap practice:** after the regression and neural-network material, run [the supplied CPU policy lab](0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-01-CPU-Policy.md). Explain the input vector, prediction, training loss and held-out result. A learned mapping from state to action in this fixture is an imitation-learning exercise; it is not an implementation of AlphaZero or evidence of pretrained robot capability.

## 2. Computer vision before advanced computational photography

Use the term-specific [16-385 Spring 2026 page](https://16385.courses.cs.cmu.edu/spring2026/) and [course information](https://16385.courses.cs.cmu.edu/spring2026/courseinfo). Topics progress from filtering and features through camera geometry, recognition and tracking. The expected background includes programming, linear algebra, calculus and basic probability; computational photography is not listed as a prerequisite. The course page's signup instructions require a code from Canvas, so do not assume every assignment or lecture link is anonymously available. The older `www.cs.cmu.edu/~16385/` landing page identifies itself as **Spring 2020**, not Fall 2024.

For a public reading route, use Szeliski's book below and the roadmap's [computer vision course](Track%20A%20-%20Hardware%20and%20Edge%20AI/3.%20Computer%20Vision/Guide.md). Start by tracing one calibrated image point into a camera ray; then study how multiple views constrain depth. Image classification alone does not recover metric position for a robot.

### 15-463 / 15-663 / 15-862: Computational Photography

The [Fall 2026 course](https://imaging.cs.cmu.edu/15-463/) is an advanced undergraduate/introductory graduate option. Its prerequisites include image computation plus programming and mathematics; 16-385 and graphics are among the listed ways to establish that background. The current offering has **six Python-and-photography assignments** and a final project, replacing the old 2018 description of seven MATLAB assignments. Camera loans and specialist optical equipment are resources for enrolled students.

Study its public imaging material for optics, exposure, HDR, calibration, lightfields and computational sensing. Reading selected image-formation lectures early is helpful; completing the entire course before introductory vision is unnecessary. For independent work, use available image datasets or your own camera and state which acquisition tasks you could actually reproduce. Processing provided images does not demonstrate sensor calibration or construction of a structured-light scanner.

The [Fall 2018 material](https://imaging.cs.cmu.edu/15-463/2018_fall/) remains an archive. Keep the term attached to any assignment or software requirement you reuse.

## 3. Graphics & imaging choices

| Your question | Useful course direction | Preparation |
|---|---|---|
| How are 3D scenes projected and rendered? | Computer Graphics, currently listed as 15-362/15-662 in the photography prerequisites | Programming, vectors and matrices |
| What can be inferred from an image? | 16-385 Computer Vision | Linear algebra, calculus, probability and programming |
| How can camera design and processing improve a measurement? | 15-463 Computational Photography | Prior image computation and the relevant physics |

These are complementary directions, not a mandatory chain of three full courses. See the [current photography prerequisites](https://imaging.cs.cmu.edu/15-463/) for course-number changes and the [SCS catalog](https://coursecatalog.web.cmu.edu/schools-colleges/schoolofcomputerscience/courses/) for formal enrollment requirements.

## 4. Self-study mapping to the roadmap

This sequence is a recommendation for independent learning, not CMU degree equivalence:

1. Learn functions, arrays, debugging and basic data structures. Merely knowing Python syntax is not the same preparation as completing 15-122.
2. Learn vectors/matrices, derivatives, conditional probability and expectations as you use them in [the foundations lessons](0.%20AI%20and%20Robot%20Learning%20Foundations/01-Math-and-Training.md).
3. Study selected 07-280 notes while training and evaluating the small CPU policy. Keep test examples out of training and model selection.
4. Study introductory vision, then choose deeper imaging/graphics material if your sensor or perception task needs it.
5. Connect perception to frames, timing and control through [the robotics course reference](../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/CMU-Robotics-Courses.md).

| Engineering outcome | Reading focus | Roadmap artifact |
|---|---|---|
| Understand a model before optimizing it | ML formulation, optimization, neural networks | Saved policy and held-out error from the CPU lab |
| Understand perception failures | Image formation, calibration, recognition | [Vision experiments](Track%20A%20-%20Hardware%20and%20Edge%20AI/3.%20Computer%20Vision/Guide.md) with declared input geometry |
| Explain stale predictions in a robot loop | Inference plus feedback and observation age | [Timed policy trace](0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-02-Timed-Robot-Loop.md) |
| Choose an accelerator workload | Model operators and their tensor shapes | [Inference optimization](../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/Guide.md) with a reference output |

CMU's AI or vision syllabus does not by itself establish a Jetson, Holoscan or ROS 2 installation path. Those software integrations belong to the corresponding roadmap labs and their pinned compatibility records.

## 5. Szeliski book and additional resources

Richard Szeliski's [Computer Vision: Algorithms and Applications, second edition](https://szeliski.org/Book/) provides an author-hosted PDF for personal use, corrections and links to related courses. Use its image-formation, geometric-transform and recognition chapters alongside 16-385. It is a reading resource, not an autograder or a replacement for image experiments.

The author's course list includes offerings from several universities and years. Follow the actual term page before adopting its software setup or expecting lectures, assignments and solutions to be public. A book homepage is not a direct link to a particular MIT, Berkeley or Michigan course.
