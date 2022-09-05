male(quentin_gerber).
male(nathaniel_hill).
male(nicholas_gerber).
male(colm_lehman).
male(roman_walsh).
male(erik_lehman).
male(jimmy_walsh).
male(ethan_walsh).
female(suzanne_rapheal).
female(alexandra_fisher).
female(stacy_hill).
female(theadosia_gerber).
female(yolanda_gerber).
female(jen_gerber).
female(gidget_Kate_lehman).
female(franny_lehman).
female(lucy_van).
female(michelle_walsh).
female(annie_lehman).
spouse(quentin_gerber, suzanne_rapheal).
spouse(nathaniel_hill, alexandra_fisher).
spouse(antony_gerber, stacy_hill).
spouse(theadosia_gerber, colm_lehman).
spouse(franny_lehman, roman_walsh).
spouse(erik_lehman, lucy_van).
parent(quentin_gerber, antony_gerber).
parent(suzanne_rapheal, antony_gerber).
parent(quentin_gerber, nicholas_gerber).
parent(suzanne_rapheal, nicholas_gerber).
parent(nathaniel_hill, stacy_hill).
parent(alexandra_fisher, stacy_hill).
parent(antony_gerber, theadosia_gerber).
parent(stacy_hill, theadosia_gerber).
parent(antony_gerber, yolanda_gerber).
parent(stacy_hill, yolanda_gerber).
parent(antony_gerber, jen_gerber).
parent(stacy_hill, jen_gerber).
parent(theadosia_gerber, gidget_Kate_lehman).
parent(colm_lehman, gidget_Kate_lehman).
parent(theadosia_gerber, franny_lehman).
parent(colm_lehman, franny_lehman).
parent(theadosia_gerber, erik_lehman).
parent(colm_lehman, erik_lehman).
parent(franny_lehman, michelle_walsh).
parent(roman_walsh, michelle_walsh).
parent(franny_lehman, jimmy_walsh).
parent(roman_walsh, jimmy_walsh).
parent(franny_lehman, ethan_walsh).
parent(roman_walsh, ethan_walsh).
parent(erik_lehman, annie_lehman).
parent(lucy_van, annie_lehman).

father(X,Y):-male(X),parent(X,Y).
mother(X,Y):-female(X),parent(X,Y).
siser(X,Y):-parent(Z,X),parent(Z,Y),female(X), X\=Y.
brother(X,Y):-parent(Z,X),parent(Z,Y),male(X), X\=Y.
son(X,Y):-parent(Y,X),male(X).
daughter(X,Y):-parent(Y,X),female(X).
grandfather(X,Y):-parent(Z,Y),father(X,Z).
grandmother(X,Y):-parent(Z,Y),mother(X,Z).
ancestor(X,Y):-parent(X,Y);(parent(X,Z),ancestor(Z,Y)).
mother_in_law(X,Y):-(spouse(Y,Z);spouse(Z,Y)),mother(X,Z).
father_in_law(X,Y):-(spouse(Y,Z);spouse(Z,Y)),father(X,Z).
