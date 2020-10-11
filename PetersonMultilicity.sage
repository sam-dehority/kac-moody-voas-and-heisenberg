# %% codecell
import numpy as np
import itertools
import functools
import sage.combinat.words.lyndon_word as lw
import re
from sage.algebras.lie_algebras.lie_algebra_element import LieBracket, LyndonBracket, GradedLieBracket, LieGenerator

def gottsche(b0,b1,b2,b3,b4, z,Q, prec=10):
    term = lambda m : (1+z^(2*m-1)*Q^m)^b1*(1+z^(2*m+1)*Q^m)^b3/((1-z^(2*m-2)*Q^m)^b0*(1-z^(2*m)*Q^m)^b2*(1-z^(2*m+2)*Q^m)^b4)
    out = prod([term(n) for n in range(1,prec+1)])
    minbetti = min([a for a in [b0,b1,b2,b3,b4] if a > 0])
    return(out.O(prec*minbetti))
k3_pic_fock = lambda Q,r,n: gottsche(1,0,r,0,1, -1, Q, n)
lattice_fock =  lambda Q,r,n: gottsche(0,0,r,0,0, -1, Q, n)


def nested_map(fn, tree):
    if tree in ZZ:
        return fn(tree)
    else:
        return list(map(lambda t : nested_map(fn, t), tree))

def nested_length(tree):
    if type(tree) == type([]):
        return sum(map(nested_length, tree))
    else:
        return 1

def unit_vect(i, n, val=1):
    a = [0]*n
    a[i] = val
    return vector(a)


# %% codecell

class LorentzianBKMRootSystem(SageObject):
    def __init__(self, cartan_matrix):
        self._cartan_mtx  = cartan_matrix
        self._rk = len(self._cartan_mtx[0])
        t = False
        for i in range(0,self._rk):
            if self._cartan_mtx[i][i] <= 0:
                t = True
        self._borcherds = t
        self._prebasis_naive_gram_matrices = {}
        self._freeE = LieAlgebra(QQ, self._rk, 'e')
        self._freeF = LieAlgebra(QQ, self._rk, 'f')
        self._LyndonE = self._freeE.Lyndon()
        self._LyndonF = self._freeF.Lyndon()
        self.serre_relations_e = self.serre_relations('e')
        self.serre_relations_f = self.serre_relations('f')
        self._v = VectorSpace(QQ,self._rk,inner_product_matrix=cartan_matrix)

    def _repr_(self):
        return("Lorentzian Root System with of rank {} with symmetric diagonal 0 or 2 cartan matrix \n{}".format(self._rk, matrix(self._cartan_mtx).__repr__()))

    def __hash__(self):
        return(hash(repr(self)))

    def isBorcherds(self):
        return(self._borcherds)
    def pair(self, a,b):
        #only allow a_ii = 2 or 0
        return(self._v(a).inner_product(self._v(b)))
    def pair_w_rho(self, a):
        out = 0
        for i in range(0,self._rk):
            if self._cartan_mtx[i][i] == 2:
                out += a[i]
        return(out)

    def serre_relations(self, e_or_f):
        r"""
        Returns a list of the serre relations defining the lie algebra
        e.g. [e_i, [e_i [... [e_i ej]]]] with 1- c_ij copies of e_i, if c_ii ==
        2 is a Serre relation. This would be the Lyndon monomial if i<j

        INPUT:

        - ``e_or_f`` -- character; whether we want the e or f relations

        OUTPUT: A list of monomials in Lyndon form, i.e. they are monomials in
        the free algebra on either e_i or f_i, of type LyndonBracket

        EXAMPLES:

        This example illustrates a simple 2x2 case ::

            sage: L = LorentzianBKMRootSystem([[2,-3],[-3,2]])
            sage: L.serre_relations('e')
            [[e0, [e0, [e0, [e0, e1]]]], [[[[e0, e1], e1], e1], e1]]

        This example shows what happens for simple imaginary roots ::

            sage: L = LorentzianBKMRootSystem([[0,-1],[-1,2]])
            sage: L.serre_relations('e')
            [[[e0, e1], e1]]

        """
        apply_n = lambda a, b, n : [a,b] if n ==1 else [a, apply_n(a, b, n-1)]
        if e_or_f == 'e':
            lyn = self._LyndonE
        else:
            lyn = self._LyndonF
        gens = lyn.gens()
        out = []
        for (i,j) in [(i, j) for i in range(self._rk) for j in range(self._rk) if self._cartan_mtx[i][i] == 2 and i != j]:
            cij = self._cartan_mtx[i][j]
            copies = 1 - cij
            out.append(lyn(apply_n(gens[i], gens[j], copies)).list()[0][0])
        return out

    @staticmethod
    def dominance_full_order(max_val,dim):
        r"""
        Returns a full order on the set of tuples with values integers less than
        a given number, which refines the order that (s1, ..., sk) <= (t1, ..., tk) iff si <= ti for all i

        INPUT:

        - ``max_val`` -- The size of the box, largest integers as coordinates
          of tuple

        - ``dim`` -- length of each tuple
        """
        if max_val == 0:
            yield (0,)*dim
        elif max_val == 1:
            for i in range(0,dim+1):
                for ones in itertools.combinations(range(0,dim), i):
                    yield tuple([1 if x in ones else 0 for x in range(0,dim)])
        else:
            for part in LorentzianBKMRootSystem.dominance_full_order(max_val-1, dim):
                yield part
            for max_places in LorentzianBKMRootSystem.dominance_full_order(1, dim):
                num_places_max = sum(max_places)
                if num_places_max == 0:
                    continue
                l = LorentzianBKMRootSystem.dominance_full_order(max_val - 1, dim - num_places_max)
                for part in l:
                    b = 0
                    y = [0]*dim
                    for j in range(dim):
                        if max_places[j] == 1:
                            y[j] = max_val
                        else:
                            y[j] = part[b]
                            b += 1
                    yield tuple(y)

    def _roots_ht_n(self, ht):
        rk = self._rk
        bars_in_stars = [sorted(s.list()) for s in Subsets(ht + rk - 1, rk-1)]
        for b in bars_in_stars:
            yield tuple([b[0] - 1 ] + [b[i] - b[i-1] -1  for i in range(1, rk-1)] + [rk + ht - b[rk-2] -1])

    def root_space_basis(self, root, dual = False, monomials_in_F = False):
        r"""
        Returns a basis for the given root space

        INPUT:

        - ``root`` -- tuple; coordiates of postiive root as
          decomposed into positive simples. Must be E!

        - ``dual`` -- If True, returns the dual basis for the opposite root space

        - ``monomials_in_F`` -- If True, when the bases are returned, the elements in E will instead
          be Lyndon elements, but those in F will be monomials.

        OUTPUT:

         A list of elements [e_{i_1}[e_{i_2} [... e_{i_n}]]] of LyndonE
         such that the total list is a basis for the root space root.


        EXAMPLES:

        This example illustrates a multiplicity 3 root space, and that the resulting bases really are dual. Note that we have to cast the e monomials as lyndon elements ::

            sage: L = LorentzianBKMRootSystem([[2,-2,0],[-2,2,-1],[0,-1,2]])
            sage: g = L.gram_matrix_on_prebasis((3,3, 1));g
            [  0   0   0   0   0   0]
            [  0  16   0   0   0 -16]
            [  0   0  32 -32   0 -32]
            [  0   0 -32  64 -16  32]
            [  0   0   0 -16   8   0]
            [  0 -16 -32  32   0  48]
            sage: es, fs = L.root_space_basis((3,3,1), dual=True); es, fs
            ([[e0, [[e0, e1], [[e0, [e1, e2]], e1]]],
             [e0, [[[e0, e1], e1], [e0, [e1, e2]]]],
             [[e0, [e0, [e1, e2]]], [[e0, e1], e1]]],
             [1/16*[f0, [[f0, f1], [[f0, [f1, f2]], f1]]],
             1/16*[f0, [[[f0, f1], f1], [f0, [f1, f2]]]] + 1/32*[[f0, [f0, [f1, f2]]], [[f0, f1], f1]],
             1/32*[f0, [[[f0, f1], f1], [f0, [f1, f2]]]] + 1/32*[[f0, [f0, [f1, f2]]], [[f0, f1], f1]]])
            sage: les = [L._LyndonE(e) for e in es]
            sage: test = Matrix(3, 3)
            sage: for i in range(3):
            sage:     for j in range(3):
            sage:         test[i, j] = L.pair_root_space_elements(fs[i], es[j])
            sage: test
            [1 0 0]
            [0 1 0]
            [0 0 1]


        NOTE::

            This returns elements of the Ei basis as monomials (i.e. you cannot add them!!) and the F_i basis as actual free algebra elements.

        """
        a_prebasis = self.root_space_prebasis(root)
        ln = len(a_prebasis)
        gm = self.gram_matrix_on_prebasis(root)
        col_basis = gm.column_module().basis()
        ech_cols = [list(b).index(1) for b in col_basis]
        a_basis = [a_prebasis[i] for i in ech_cols]
        if dual:
            rk = len(a_basis)
            restricted_gm = column_matrix([gm.columns()[i] for i in ech_cols])
            row_basis = restricted_gm.column_module().basis()
            indices_image_of_restricted = [list(b).index(1) for b in row_basis]
            a_minus_basis = [self.root_space_prebasis(-vector(root))[i] for i in indices_image_of_restricted]
            gm_on_basis = Matrix([restricted_gm[i] for i in indices_image_of_restricted]) # matrix from rows
            change_of_basis_mx = ~gm_on_basis
            if not monomials_in_F:
                # uses e_j GM^-1 GM e_i = δ_ij so e_j GM^-1 is dual to e_i
                row_to_F_elt = lambda rs : sum(rs[i]* self._LyndonF(a_minus_basis[i]) for i in range(rk))
                return((a_basis, [row_to_F_elt(r) for r in change_of_basis_mx.rows()]))
            else:
                # uses e_jGM GM^-1  e_i = δ_ij so GM^-1e_i is dual to e_j
                column_to_E_elt = lambda cs : sum(cs[i]* self._LyndonE(a_basis[i]) for i in range(rk))
                return(([column_to_E_elt(r) for r in change_of_basis_mx.columns()], a_minus_basis))
        else:
            return a_basis

    @cached_method
    def gram_matrix_on_prebasis(self, root):
        r"""
        Computes the gram matrix of the pairing between the given root space which is positive and its dual, normalized so (f_i|e_i) = 1

        INPUT:

        - ``root`` -- the root of e_alpha, i.e. the positive root.


        OUTPUT: matrix G of pairings g_ij = (f_I | g_J)

        EXAMPLES:

        This example illustrates a 2d root space ::

            sage: L = LorentzianBKMRootSystem([[2,-2,0],[-2,2,-1],[0,-1,2]])
            sage: L.gram_matrix_on_prebasis((2,2,1))
            [4 0]
            [0 8]


        """
        a_minus_basis = self.root_space_prebasis(-vector(root))
        a_basis = self.root_space_prebasis(root)
        ln = len(a_basis)
        gm = [[0 for i in range(ln)] for j in range(ln)]
        for i in range(ln):
            for j in range(ln):
                gm[i][j] = self.pair_root_space_elements(a_minus_basis[i], a_basis[j])
        return(matrix(gm))

    def _zero_by_serre(self, x):
        #takes a Lyndon monomial and returns True if it contains a serre
        #relation
        if isinstance(x, LieGenerator):
            return False
        e_or_f = x.to_word()[0][0]
        if e_or_f == 'e':
            rels = self.serre_relations_e
        else:
            rels = self.serre_relations_f
        if x in rels:
            return True
        else:
            return self._zero_by_serre(x[0]) or self._zero_by_serre(x[1])
    def _root_space_prebasis_no_serre(self, root):
        rvec = vector(root)
        if sum(rvec) > 0:
            a =  [lw.standard_bracketing(u) for u in LyndonWords(list(root))]
            gen_from_int = lambda t : self._LyndonE.gens()[t - 1]
            b = nested_map(gen_from_int, a)
            c = list(map(self._LyndonE, b))
            return c
        elif sum(rvec) < 0:
            a =  [lw.standard_bracketing(u) for u in LyndonWords(list(-rvec))]
            gen_from_int = lambda t : self._LyndonF.gens()[t - 1]
            b = nested_map(gen_from_int, a)
            c = list(map(self._LyndonF, b))
            return c
    def root_space_prebasis(self, root):
        r"""
        Gives a set of free algebra elements spanning the given root space

        INPUT:

        - ``root`` -- tuple specifying the root


        OUTPUT: list of elements in free algebra generated by e_i or f_i

        EXAMPLES:

        This example illustrates larger than 1 multiplicity possibly ::

            sage: L = LorentzianBKMRootSystem([[2,-2,0],[-2,2,-1],[0,-1,2]])
            sage: L._root_space_prebasis((2,2,1))
            [[e0, [[e0, [e1, e2]], e1]], [[e0, e1], [e0, [e1, e2]]]]

        """
        rvec = vector(root)
        if sum(rvec) > 0:
            a =  [lw.standard_bracketing(u) for u in LyndonWords(list(root))]
            gen_from_int = lambda t : self._LyndonE.gens()[t - 1]
            b = nested_map(gen_from_int, a)
            c = list(map(self._LyndonE, b))
            d = [w for w in c if not self._zero_by_serre(w.list()[0][0])]
            return d
        elif sum(rvec) < 0:
            a =  [lw.standard_bracketing(u) for u in LyndonWords(list(-rvec))]
            gen_from_int = lambda t : self._LyndonF.gens()[t - 1]
            b = nested_map(gen_from_int, a)
            c = list(map(self._LyndonF, b))
            d = [w for w in c if not self._zero_by_serre(w.list()[0][0])]
            return d

    def lyndon_root_space(self, x):
        r"""
        Returns the root space of a lyndon word,
        interpreted as an element of the lie algebra

        INPUT:

        - ``x`` -- lyndon basis element monomial; either in self._LyndonE or
          self._LyndonF, but not actually these elements, their monomials, attanable as elt.list()[0][0]

        OUTPUT: the tuple corresponding to the root space x lies in. Negative
        root for f, positive for e.

        EXAMPLES:

        We calculate this for an easy element ::

            sage: L = LorentzianBKMRootSystem([[2,-2,0],[-2,2,-1], [0,-1,2]])
            sage: e0,e1,e2 = L._freeE.gens()
            sage: x = L._LyndonE([e0, [e2, e0]]); x
            -[e0, [e0, e2]]
            sage: L.lyndon_root_space(x.list()[0][0])
            (2,0,1)

        """
        e_or_f = x.to_word()[0][0]
        genEmonomials = list(map(lambda g : g.list()[0][0], self._LyndonE.gens()))
        genFmonomials = list(map(lambda g : g.list()[0][0], self._LyndonF.gens()))
        if x in genEmonomials and e_or_f == 'e':
            return self._v(unit_vect(genEmonomials.index(x), self._rk))
        elif x in genFmonomials and e_or_f == 'f':
             return self._v(unit_vect(genFmonomials.index(x), self._rk, val = -1))
        else:

            return self.lyndon_root_space(x._left) + self.lyndon_root_space(x._right)
    def pair_root_space_elements(self, fw, ew):
        r"""
        Returns the invariant form paired on the two elements fw and ew. Used to construct a basis in terms of lyndon LyndonWords

        INPUT:

        - ``fw`` -- Element of self._LyndonF; i.e. a element in the free algebra
          generated by the fi

        - ``ew`` -- Element of self._LyndonE); i.e. element of free algebra
          generated by the ei

        OUTPUT: the pairing (f | e)

        EXAMPLES:

        This example illustrates a basic pairing ::

            sage: L = LorentzianBKMRootSystem([[2,-2,0],[-2,2,-1], [0,-1,2]])
            sage: e0,e1,e2 = l._LyndonE.gens()
            sage: f0,f1,f2 = l._LyndonF.gens()
            sage: x  = l._LyndonE([e0,[e1,e2]])
            sage: y  = l._LyndonF([f0,[f1,f2]])
            sage: l.pair_root_space_elements(y, x)
            2

        """
        fs = fw.list() #gets the monomial corresponding to fw so we may
        #get the terms its formed from. Without this from [a,b] we could not
        #get a or b.
        es = ew.list()
        if fs == [0] or es == [0]:
            return 0
        else:
            return sum([c1*c2* self._pair_root_space_monomials(fm, em) for (em, c1) in es for (fm, c2) in fs])
    @functools.lru_cache(maxsize=None)
    def _pair_root_space_monomials(self, fw, ew):
        #Only to be called form pair_root_space_elements
        alphaf = self.lyndon_root_space(fw)
        alphae = self.lyndon_root_space(ew)
        if alphaf + alphae != 0:
            s = alphaf + alphae
            return 0
        elif sum(alphae) == 1:
            return 1
        else:
            #We compute ([f_w1, f_w2] | [e_w3, e_w4]) by moving the shortest
            #word over to the other side using invariance of the pairing
            # expanding the bracktet by repeated use of
            # 1) the Jacobi identity
            # 2) [e_i, f_j] = d_ij h_i
            # 3) [h_alpha, e_beta] = <alpha, beta> e_beta
            # which reduces the computation to pairings on shorter elements
            fw1 = fw._left
            fw2 = fw._right
            ew3 = ew._left
            ew4 = ew._right
            w1_len = len(fw1.to_word())
            w2_len = len(fw2.to_word())
            w3_len = len(ew3.to_word())
            w4_len = len(ew4.to_word())
            min_length = min(w1_len, w2_len, w3_len, w4_len)
            E = lambda t : self._LyndonE({t:1})
            F = lambda t : self._LyndonF({t:1})
            zero = tuple([0]*self._rk)
            if w1_len == min_length:
                #
                # [fw1, ew3] and [fw1, ew4] are both generated by es. Check this by assertion

                h1_new, f1_new, e1_new = self._bracket_fe(
                            self._bracket_fe_monomials(fw1, ew3)
                            , (zero,0, E(ew4))
                        )
                assert h1_new == 0 , "bracket should be only e"
                assert f1_new == 0, "bracket should be only e"
                h2_new, f2_new, e2_new = self._bracket_fe(
                            (zero, 0, E(ew3)),
                            self._bracket_fe_monomials(fw1, ew4)
                        )
                assert h2_new == 0 , "bracket should be only e"
                assert f2_new == 0, "bracket should be only e"
                t1 = self.pair_root_space_elements(F(fw2), e1_new )
                t2 = self.pair_root_space_elements(F(fw2), e2_new )
                return (-t1 - t2)
            elif w2_len == min_length:
                e1_new = self._bracket_fe(
                            self._bracket_fe_monomials(fw2, ew3),
                            (zero, 0, E(ew4))
                        )[2]
                e2_new = self._bracket_fe(
                            (zero, 0, E(ew3)),
                            self._bracket_fe_monomials(fw2, ew4)
                        )[2]
                t1 = self.pair_root_space_elements(F(fw1), e1_new )
                t2 = self.pair_root_space_elements(F(fw1), e2_new )
                return (t1 + t2)
            elif w3_len == min_length:
                # paring = ([[fw1, ew3], fw2] + [ fw1, [fw2 , ew3 ]] | ew4 )
                f1_new = self._bracket_fe(
                            self._bracket_fe_monomials(fw1, ew3),
                            ( zero, F(fw2), 0)
                        )[1]
                f2_new = self._bracket_fe(
                            (zero, F(fw1), 0),
                            self._bracket_fe_monomials(fw2, ew3)
                        )[1]
                t1 = self.pair_root_space_elements(f1_new, E(ew4))
                t2 = self.pair_root_space_elements(f2_new, E(ew4))
                return (t1 + t2 )
            else:
                f1_new = self._bracket_fe(
                            self._bracket_fe_monomials(fw1, ew4),
                            (zero, F(fw2), 0)
                        )[1]
                f2_new = self._bracket_fe(
                            (zero, F(fw1), 0),
                            self._bracket_fe_monomials(fw2, ew4)
                        )[1]
                t1 = self.pair_root_space_elements(f1_new, E(ew3))
                t2 = self.pair_root_space_elements(f2_new, E(ew3))
                return ( -t1 - t2 )
    def _bracket_fe_monomials(self, f, e):
        ###
        # input, two tuples or lyndon monomials
        # output, (h, f, e) representing h+l where h is a tuple correpsonding to
        # a cartan subalgebra element, f is a lyndon element generated by fi, e is lyndon generated by ei
        ###
        zero = tuple([0]*self._rk)
        fv = not (isinstance(f,LieGenerator) or isinstance(f, LieBracket))
        ev = not (isinstance(e,LieGenerator) or isinstance(e, LieBracket))
        if fv and ev:
            return (zero, 0, 0)
        elif fv:
            return ( zero, 0,  self.pair(f, self.lyndon_root_space(e)) * self._LyndonE({e:1}))
        elif ev:
            return  (zero, - self.pair(self.lyndon_root_space(f), e) * self._LyndonF({f:1}), 0)
        else:
            #break up the long one using Jacobi, keep going
            # [[fw0 , fw1], e] = [[fw0, e], fw1] + [fw0, [fw1, e]]
            f_len = len(f.to_word())
            e_len = len(e.to_word())
            if f_len > e_len:
                fw0 = f[0]
                fw1 = f[1]
                h1, f1, e1 = self._bracket_fe(self._bracket_fe_monomials(fw0,e)
                                       ,(zero, self._LyndonF({fw1:1}), 0))
                h2, f2, e2 = self._bracket_fe((zero, self._LyndonF({fw0:1}), 0),
                                        self._bracket_fe_monomials(fw1, e))
                return(tuple(h1+h2), f1 + f2, e1+e2 )
            elif f_len == e_len == 1:
                alphaf = self.lyndon_root_space(f)
                alphae = self.lyndon_root_space(e)
                if alphaf == -alphae :
                    return (tuple(alphaf), 0,0)
                else:
                    return (zero, 0,0)
            else:
                # e_len >= f_len, e_len > 1
                # use [f, [e0, e1]] = [[f, e0], e1] + [e0 , [f, e1]]
                ew0 = e[0]
                ew1 = e[1]
                h1, f1, e1 = self._bracket_fe(self._bracket_fe_monomials(f,ew0)
                                       ,(zero, 0, self._LyndonE({ew1:1})))
                h2, f2, e2 = self._bracket_fe((zero, 0, self._LyndonE({ew0:1})),
                                        self._bracket_fe_monomials(f, ew1))
                return(tuple(h1+h2), f1 + f2, e1+e2 )
    @functools.lru_cache(maxsize=None)
    def _bracket_fe(self, x, y):
        r"""
        returns the lie bracket of two triangularly decomposed elements

        INPUT:

        - ``x`` -- tuple (h1,f1, e1); h1 is a TUPLE reprsenting a cartan
          element, f1 is in self._LyndonF, e1 is in self._LyndonE

        - ``y`` -- tuple; just like x

        OUTPUT: tuple (h,f,e) with h a VECTOR, f,e as above in Lyndon free
                algebras

        EXAMPLES:

        This example illustrates a bracket ::

            sage: L = LorentzianBKMRootSystem([[2,-2,0],[-2,2,-1], [0,-1,2]])
            sage: e0,e1,e2 = l._LyndonE.gens()
            sage: f0,f1,f2 = l._LyndonF.gens()
            sage: L._bracket_fe( ((0,0,0),0, 2*e1), ((0,0,0), f1, 0) )
            ((0,2,0), 0, 0)

        NOTE ::

        Both x and y must be hashable in order for the lru_cache to work and significantly speed up the calculation of these bases. Thus the first components are tuples but not vectors. On the other hand, this returns a triple where the first component is a vector.
        """
        ###
        #input, two tuples (h1,f1, e1), (h2, f2, e2) as in _bracket_fe_monomials output
        # h1 and h2 are TUPLES, i.e. immutable for cache
        # output tuple (h, f ,e ), their bracket [h1 + f1 + e1, h2 + f2 + e2]
        # h is a VECTOR
        # can feed _bracket_fe_monomials to bracket_fe
        # to access result, known to lie in non-zero root space, take [1]
        ###
        h1, f1, e1 = vector(x[0]), x[1], x[2]
        h2, f2, e2 = vector(y[0]), y[1], y[2]
        f1s = f1.list()
        e1s = e1.list()
        f2s = f2.list()
        e2s = e2.list()
        h1z, h2z, f1z, f2z, e1z, e2z = (h1 ==0, h2 ==0,
                                        f1s ==[0], f2s == [0],
                                        e1s == [0], e2s == [0])
        f1s = f1s if not f1z else []
        e1s = e1s if not e1z else []
        f2s = f2s if not f2z else []
        e2s = e2s if not e2z else []

        zero = vector([0]*self._rk)
        vector_first = lambda a : (vector(a[0]), ) + a[1:]
        z3 = (zero, 0, 0)
        def tsum(a,b):
            #print(f"summing {a} and {b}")
            return (a[0] + b[0],
                     self._LyndonF(a[1]) + self._LyndonF(b[1]),
                     self._LyndonE(a[2])+ self._LyndonE(b[2]))
        tneg = lambda a : (-1*a[0], -a[1], -a[2])
        tscale = lambda c, a : (c*a[0], c*a[1], c*a[2])
        h1h2 = (zero, 0, 0)
        h1e2 = z3 if h1z or e2z else (zero, 0, sum([c*(self._bracket_fe_monomials(h1, ew)[2])  for (ew, c) in e2s ]))
        h1f2 = z3 if h1z or f2z else (zero, -sum([c*(self._bracket_fe_monomials(fw, h1)[1])  for (fw, c) in f2s ]), 0)
        f1h2 = z3 if f1z or h2z else (zero, sum([c*(self._bracket_fe_monomials(fw, h2)[1])  for (fw, c) in f1s ]), 0)
        f1f2 = (zero, self._LyndonF([f1, f2]), 0)
        f1e2 = z3 if f1z or e2z else reduce (tsum,
                       [tscale(c*d, vector_first(self._bracket_fe_monomials(fw, ew)) )
                         for (ew,c) in e2s for (fw,d) in f1s
                       ],
                       (zero, 0, 0)
                       )
        e1h2 = z3 if e1z or h2z else (zero,0, -sum([c*(self._bracket_fe_monomials(h2, ew)[2])  for (ew, c) in e1s ]))
        e1f2 = z3 if e1z or f2z else tneg( reduce (tsum,
                       [tscale( c*d, vector_first( self._bracket_fe_monomials(fw, ew)))
                         for (ew,c) in e1s for (fw,d) in f2s
                       ],
                       (zero,0, 0)
                       ))
        e1e2 = (zero, 0, self._LyndonE([e1,e2]))
        return (reduce(tsum, [h1h2, h1e2, h1f2, f1h2, f1f2, f1e2, e1h2, e1f2, e1e2], (zero, 0, 0)))

    def _peterson_mult_table(self, box_size):
        r"""
        Returns a table which allows us to extract the root multiplicity
        of root in self

        INPUT:

        - ``box_size`` -- integer; how big the table is (-1)

        OUTPUT: a table where we can access mult by passing a positive root as a tuple

        EXAMPLES:

        This example illustrates calculating a non-trivial imaginary root
        mutliplicity ::

            sage: L = LorentzianBKMRootSystem([[2,-2,0],[-2,2,-1], [0,-1,2]])
            sage: pmt = L._peterson_mult_table(box_size= 4)
            sage: pmt[(3,3,1)]
            3

        """
        # mlist is a rank(L) list and we compute c_beta
        # for simple roots beta with beta<mlist under the obvious
        # partial order
        # based on formula c_beta = 1/(beta| beta - 2 rho) \sum_{beta' + beta'' = beta} (beta'| beta'') c_beta' c_beta'
        dim = self._rk
        c = np.full([box_size]*dim,-1.0)
        m = np.full([box_size]*dim,0.0)
        c[(0,)*dim] = 0
        m[(0,)*dim] = 0
        for i in range(self._rk):
            tup = (0,)*i + (1,) + (0,)*(self._rk-i-1)
            c[tup] = 1
            m[tup] = 1
        for beta in LorentzianBKMRootSystem.dominance_full_order(box_size-1, dim):
            #print('beta = %s' %(beta,))
            norm_beta = self.pair(beta, beta)
            beta_dot_rho = self.pair_w_rho(beta)
            biggest = max(beta)
            if sum(beta) <= 1:
                continue
            if norm_beta > 2:
                m[beta] = 0
                c[beta] = 0
                for n in range(1,biggest+1):
                    if all([a%n == 0 for a in beta]):
                        beta_n = map(lambda z: z/n, beta)
                        c[beta] += m[tuple(beta_n)]/n
                        #print('n = %s, b_n = %s, c[beta] = %s, m[beta_n]/n = %s' % (n, beta_n, c[beta], m[beta_n]/n))
                assert(c[beta]>= 0)
                #print('case 1 b = %s, |b| = %s, c[b] = %s m[b] = %s' % (beta, norm_beta, c[beta], m[beta]))
            elif len([a for a in beta if a > 0]) == 1:
                s = sum(beta)
                c[beta] = 1/s
                m[beta] = 1 if s == 1 else 0
                assert(c[beta]>= 0)
                #print('case 2 b = %s, |b| = %s, c[b] = %s m[b] = %s, sum(b) = %s' % (beta, norm_beta, c[beta], m[beta], s))
            else:
                total = 0
                for beta_p in LorentzianBKMRootSystem.dominance_full_order(biggest,dim):
                    if beta_p == (0,)*dim:
                        continue
                    if any([x < y for (x,y) in zip(beta, beta_p)]):
                        continue
                    if beta_p == beta:
                        break
                    beta_pp = tuple(np.subtract(beta, beta_p))
                    #print('b = %s, bp = %s, bpp = %s' %(beta, beta_p, beta_pp))
                    #print('c[bp] = %s, c[bpp] = %s' %(c[beta_p], c[beta_pp]))
                    if (c[beta_p] < 0 or c[beta_pp] < 0):
                        raise ValueError('c < 0 when b = %s, bp = %s, bpp = %s, c[bp] = %s, c[bpp = %s], m[bp] = %s, m[bpp] = %s' %(beta, beta_p, beta_pp, c[beta_p], c[beta_pp], m[beta_p], m[beta_pp]))
                    assert(c[beta_p] >= 0)
                    assert(c[beta_pp] >= 0)
                    total += self.pair(beta_p, beta_pp)*c[beta_p]*c[beta_pp]
                #print("b = %s, (b.b) = %f, (b.r) = %f c[beta]*() = %f " %(beta, norm_beta, beta_dot_rho, total ))
                c[beta] = 1/(norm_beta - 2*beta_dot_rho)*total
                m_total = c[beta]
                biggest = max(beta)
                for n in range(2,biggest+1):
                    if all([a%n == 0 for a in beta]):
                        beta_n = map(lambda z: z/n, beta)
                        if( beta == (5,4)):
                            print('m_total = %s, n = %s, bn = %s, m[bn] = %s' %(m_total, n, beta_n, m[beta_n]))
                        m_total -= m[tuple(beta_n)]/n
                m[beta] = m_total
                #print('case 3 b = %s, |b| = %s, c[b] = %s m[b] = %s' % (beta, norm_beta, c[beta], m[beta]))
        return(m.astype(int))

# %% codecell
class BasicRepresentation(SageObject):
    """
    Representation of the basic representation of the affinization of a LKBM algebra given by its root system.
    """
    def __init__(self, algebra):
        r"""
        Does nothing other than wrap an LBKM root system.

        INPUT:

        - ``algebra`` -- A LBKM root system
        """
        self.algebra = algebra
        self.rk = algebra._rk

    def _repr_(self):
        return "Basic representation of {}".format(self.algebra._repr_())

    def weight_mult(self, c1, ch2):
        r"""
        Returns the multiplicity of a weight in the basic representation

        INPUT:

        - ``c1`` -- tuple; The vector in the root system of the underlying
          algebra (i.e. non-affinized algebra) giving the corresponding component of the weight

        - ``ch2`` -- integer; Negative the eigenvalue of L_0 on this weight
          space, equals eigenvalue of d

        OUTPUT: the dimension of the weight space as an integer

        EXAMPLES:

        This example illustrates that the hilbert scheme of 0 points
        is a point, so has cohomology one dimensional ::

            sage: l = LorentzianBKMRootSystem([[2,-2,0],[-2,2,-1], [0,-1,2]])
            sage: F = BasicRepresentation(l)
            sage: F.weight_mult((0,0,0),0)
            1
            sage: F.weight_mult((0,0,0),2)
            0
            sage: F.weight_mult((0,0,0), -2)
            20

        Now we demonstrate that if ch2 = -1/2<c1,c1> the weight space is 1d ::

            sage: l = LorentzianBKMRootSystem([[2,-2,0],[-2,2,-1], [0,-1,2]])
            sage: F = BasicRepresentation(l)
            sage: l.pair((2,3,4),(2,3,4))
            10
            sage: F.weight_mult((2,3,4),-5)
            1

        NOTE ::

        The pairing on the root system is the negative of the pairing on chern classes, and this method usees the pairing on the root system.
        """
        n = Integer(Integer(1)/Integer(2) * (-self.algebra.pair(c1, c1))) - ch2
        if n < 0:
            return 0
        else:
            R.<q> = PowerSeriesRing(QQ, default_prec=n+2)
            fock_dim = lattice_fock(q, self.rk,n+2).coefficients()
            return fock_dim[n]

    def _pair_wts(self,l , c1,ch2,lp,c1p,ch2p):
        r"""
        These weights have level l = 1, so their pairing is given by this formula
        <l,c1,ch2| l', c1', ch2'> = <c1,c1'> +l(ch2 + ch2') = <c1,c1'> +ch2 + ch2'. Here we assume l' = 0!!!
        """
        return self.algebra.pair(c1,c1p) +lp*ch2 +l*ch2p
    def casimir_summand_tr(self, wt, alpha, k):
        r"""
        Returns the trace of e_{-\alpha}(-k) e_\alpha(k) on a weight space where
        (e_-\alpha|e_alpha) = 1 and (|) is the symmetric invariant form

        INPUT:

        - ``wt`` -- tuple; First component is coordinates of non-affinized
          component of wt, second component is eigenvalue of d

        - ``alpha`` -- tuple of integers; Root in non-affinized root system

        - ``k`` -- integer; Corresponds to the element e_\alpha \otimes t^k in
          the loop algebra, i.e. k is d component of weight \alpha

        OUTPUT: the trace of e_{-\alpha}(-k) e_\alpha(k) on a weight space on weight space wt

        EXAMPLES:

        This example illustrates ... ::

            sage: l = LorentzianBKMRootSystem([[2,-2,0],[-2,2,-1], [0,-1,2]])
            sage: F = BasicRepresentation(l)
            sage: F.casimir_summand_tr(((1,1,1), -10), (1,0,0),3)
            1112

        NOTE ::

        This is calculated using a formula analogous to the argument deriving
         Freudenthal's formula. c.f. any proof of this.
        """
        c1 = wt[0]
        ch2 = wt[1]
        d =1
        tr = 0
        j = 1
        if self._pair_wts(0,alpha, k,0, alpha, k) == 0 and self._pair_wts(1, c1, ch2,0, alpha, k) == 0:
            return 0
        while d != 0:
            jalpha = j*vector(alpha)
            c1_plus_ja = vector(c1) + jalpha
            pairing = self._pair_wts(1, c1_plus_ja, ch2+j*k,0,  alpha, k)
            dim = self.weight_mult(c1_plus_ja, ch2 +j*k)
            tr += pairing*dim
            j += 1
            d = dim
        return tr



# %% codecell

class IntegrableHighestWeightReps(Parent):
    def __init__(self, root_system):
        self.root_system = root_system
        Parent.__init__(self)
    
    def fundamental_weights(self):
        c = self.root_system._cartan_mtx
        rk = c.dimensions()[0]
        x = [0]*rk
        for i in range(rk):
            x[i] = var('x' + str(i))
        v = vector(x)
        out = [0]*rk
        for i in range(rk):
            eqs = [c.rows()[i]*v == 1] + [c.rows()[j]*v == 0 for j in range(rk) if j != i]
            dct = solve(eqs, x, solution_dict=True)[0]
            out[i] = vector(dct.values())
        return out
        
    
class IntegrableHighestWeightRep:
    def __init__(self, algebra, wt):
        """
        Class for a highest weight rep of algebra with hgihest weight
        wt. Only implements character calculation using
        Freudenthal's formula.
        
        Wt is written in basis of fundamental weights, provided by IntegraleHighsetWeightReps
        """
        self.root_system = algebra
        self.highest_wt = wt
        
    def wts(self, box_size = 10):
        r"""
        Returns a table of the multiplicity of weights in the representation

        INPUT:

        - ``box_size`` -- integer (default, 10); determines how many weights we want to compute

        OUTPUT: An array which takes as index a positive root α and returns 
        mult_L(Λ)(Λ - α), the multiplicity of the weight Λ - α in the integrable
        highest rep L(Λ) with highest weight Λ. 

        EXAMPLES:

        This example illustrates a multiplicity calculation relevant 
        to the banana configuration of curves ::

            sage: l = LorentzianBKMRootSystem([[2,-2,-2],[-2,2,-2], [-2,-2,2]])

        NOTE ::
        
        1. This calculation is done using Freudenthal's formula. See Kac 'Infinite Dimensional Lie Algebras'
        3rd. ed. problem 11.15. In particular it depends heavily on the root multplicity given by 
        LorentzianBKMRootSystem._peterson_mult_table
        
        2. Explicitly the formula is 
        
        ((Λ + ρ| Λ + ρ) - (λ + ρ | λ + ρ))dim V_λ = 2 sum_Δ+ sum_j\ge 1 mult(α)(λ + jα | α) dim V_{λ + jα}
        
        and we use that
        
        (Λ + ρ| Λ + ρ) - (λ + ρ | λ + ρ) = Λ^2 - λ^2 + 2(Λ - λ| ρ) = 2(Λ|α) - (α|α) + 2(α| ρ)
        where Λ - λ = α is now in Δ_+
        
        3. To write the indices of the output as elements of the weight lattice we must convert
        Λ - r to an element of the weight lattice for a root r. 
        
        """
        g = self.root_system
        pmt = g._peterson_mult_table(box_size = box_size)
        dim = g._rk
        hw = self.highest_wt
        reps = IntegrableHighestWeightReps(g)
        fun_wts = reps.fundamental_weights()
        Λ = sum(fun_wts[i]*hw[i] for i in range(dim)) # writes Λ in appropriate coords for pairing with roots
        hw_sqd = g.pair(Λ, Λ)
        prefactor = lambda α: -g.pair(α, α) + 2*g.pair(Λ, α) +  2*g.pair_w_rho(α)
        m = np.full([box_size]*dim,0.0)#initialize zero array
        m[(0,)*dim] = 1 # highest weight has multiplicity 1
        dom = lambda r1, r2 : all(x >=y for x,y in zip(r1, r2))
        def check_dim(α):
            if not dom(α, (0,)*dim):
                return(0)
            else:
                return(m[α])
        rts = LorentzianBKMRootSystem.dominance_full_order(box_size-1, dim)
        next(rts) #skips root (0,0,0) which is actually not a root
        for r in rts:
            if pmt[r] == 0:
                #print(f"pmt[r] = 0 for r = {r}")
                m[r] == 0
            else:
                λ = Λ - vector(r)
                less_than_r = [a for a in LorentzianBKMRootSystem.dominance_full_order(max(r), dim) if a != (0,)*dim and dom(r,tuple(a))]
                #print(f"Λ = {Λ}, λ = {λ}, r = {r}, pmt[r] = {pmt[r]}, pre = {prefactor(r)}")
                if prefactor(r) == 0:
                    m[r] = 0
                else:
                    pre = 2/prefactor(r)
                    this_wt_mult = 0
                    for α in less_than_r:
                        j = 1
                        d = 1
                        multα = pmt[α]
                        while d > 0:
                            wt_current = tuple(vector(r) - j*vector(α))#minus sign since this checks Λ - root on root
                            d = check_dim(wt_current)
                            this_wt_mult += multα*g.pair(λ + j*vector(α), vector(α))*d
                            j += 1
                    m[r] = this_wt_mult
        recursive_toint = lambda m : [recursive_toint(mi) for mi in m] if isinstance(m, list) else Integer(m)
        return(recursive_toint(m.tolist()))
        
        
        
        

