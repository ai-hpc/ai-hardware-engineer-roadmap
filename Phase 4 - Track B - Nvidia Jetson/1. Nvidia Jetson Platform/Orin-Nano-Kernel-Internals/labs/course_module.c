// SPDX-License-Identifier: GPL-2.0-only
/* Build-only teaching module: no hardware access and no runtime test claimed. */
#include <linux/init.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/errno.h>

static unsigned int sample_rate_hz = 100;
module_param(sample_rate_hz, uint, 0444);
MODULE_PARM_DESC(sample_rate_hz, "Teaching parameter in the range 1..1000");

static int __init course_module_init(void)
{
    if (sample_rate_hz < 1 || sample_rate_hz > 1000)
        return -EINVAL;
    pr_info("course_module: teaching parameter %u Hz; no hardware configured\n",
            sample_rate_hz);
    return 0;
}

static void __exit course_module_exit(void)
{
    pr_info("course_module: exit\n");
}

module_init(course_module_init);
module_exit(course_module_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Build-only course module with a bounded read-only parameter");
