.. _guide:

<Name of your check>
====================

Description
-----------

This section gives a **short** overview. It's the 30 seconds pitch
for your check and answers the following questions:

1. What is it and why should a user want to use it?
2. When is it applicable (optional)?
3. What conclusions can a user draw from its result?

Example
-------

This section provides a complete and **self-contained** example of
how to use your check.

Code
++++

::

    from pprint import pprint

    # highlight if a user has to replace sth. by using <brackets>
    data_set_1 = '<path to first data set>'
    data_set_2 = '<path to second data set>'

    def example_function(test):
        s = """
        Always provide a self-contained source code example
        containing all required imports etc.
        """
        pprint(s)

Start with **copy-pastable** source code that a user can execute on her
own computer to reproduce the exact same result. You can assume
that she already installed morpheus and thus can import our package.
Always use one of the example data sets introduced in :ref:`checks`.

Walk the user through the code and link her to the respective
api documentation.

Result
++++++

::

    Show the result of executing the code from above. Truncate the
    output if its too long. However, always show enough to be able
    to communicate to the user how she can interpret the result.
    If you truncated, highlight this by three dots on the last line
    ...

Interpretation
++++++++++++++

Show the user how to interpret the result from above. A user should be
able to view any possible result from executing your check side-by-side
with the text from this subsection to interpret the result. This is
probably the most important section of your entire documentation.

Parameters
----------

This section is optional. Use it to discuss parameters of your check. Guide
the user to pick reasonable values for those parameters.

Implementation
--------------

This section is intended for interested readers. A user should be able to
use your check without having to read this section.

Algorithm
+++++++++

Use this subsection to describe how your check processes its input to produce
its output. If applicable you can link to scientific publications.

Notes
+++++

Use this subsection to include notes about your implementation:
design decisions, why you favored one algorithm over another, ...

References
----------

Reference section.