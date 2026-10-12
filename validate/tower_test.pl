# geometric-tower test: per check, path length ell, kernel size |g_v|, D2 = ||S(v)||, tower along the step direction, ratios, and d log|g| / d ell
my ($file) = @ARGV; open(my $fh, '<', $file) or die; my (%ell, $first, %rows);
while (<$fh>) { chomp;
  if (/^(\d+) (\S+)\s+(\S+)\s+(\S+)\s+(\S+)/) { $ell{$1} = $4; next }
  if (/^(v\d+)\S* (\d+) \|/) { my ($nm,$s)=($1,$2); $first //= $nm; next unless $nm eq $first; next if $rows{$s};
    my @p = split /\|\|/; my @g = split " ", (split /\|/, $p[1])[0]; my ($tw) = $_ =~ /\|\| tower\s+(\S+\s+\S+\s+\S+\s+\S+)/; my @t = split " ", $tw;
    $rows{$s} = [$g[0], $g[4], @t]; }
}
printf "%-6s %6s %9s %9s %6s | %9s %9s %9s %9s | %6s %6s %6s | %7s\n", "step","ell","|g_v|","D2","D2/|g|","f'[u]","f''[u]","f'''[u]","f''''[u]","r21","r32","r43","dlog|g|/dell";
my ($pl,$pg);
for my $s (sort {$a<=>$b} keys %rows) { my ($g,$d2,@t) = @{$rows{$s}}; my $l = $ell{$s} // 0;
  my $slope = (defined $pl && $l > $pl) ? log($g/$pg)/($l-$pl) : 0;
  printf "%-6d %6.2f %9.2e %9.2e %6.1f | %9.2e %9.2e %9.2e %9.2e | %6.1f %6.1f %6.1f | %7.2f\n", $s,$l,$g,$d2,$d2/$g,@t, $t[1]/$t[0], $t[2]/$t[1], $t[3]/$t[2], $slope;
  ($pl,$pg) = ($l,$g); }
